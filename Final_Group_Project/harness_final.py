import pandas as pd
import numpy as np
import sys, getopt
import optparse
import pickle
from pandas import DateOffset

data_dir = './data/'

import warnings
warnings.filterwarnings("ignore")

def _calc_fin_ratio(df):
    # Leverage Ratio
    out_cols =["debt_st_to_eqty_tot", 
               "eqty_tot_to_asst_tot",
               "debt_st_to_deb_tot",
               "debt_st_to_ebitda",
               "liq_curr_ratio",
               "liq_quick_ratio",
               "wc_net_to_asst_tot",
               "asst_tang_fixed_to_asst_tot",
               "asst_intang_fixed_to_asst_tot",
               "liq_cfo_ratio",
               "ebitda_to_asst_tot",
               "prof_operations_to_asst_tot",
               "ebitda_to_exp_financing",
               "AR_to_liab_tot"]
    
    df["debt_st_to_eqty_tot"] = (df["debt_st"] / df["eqty_tot"]).where(df["debt_st"]!=0, 0)
    df["eqty_tot_to_asst_tot"] = df["eqty_tot"] / df["asst_tot"]
    df["debt_st_to_deb_tot"] = (df["debt_st"] / (df["debt_st"] + df["debt_lt"])).where(df["debt_st"]!=0, 0)
    df["debt_st_to_ebitda"] = (df["debt_st"] / df["ebitda"]).where(df["debt_st"]!=0, 0)
    # Liquidity Ratio
    df["liq_curr_ratio"] = (df["asst_current"] / df["liab_tot"]).where(df["asst_current"]!=0, 0) # we don't have current asset or liability
    df["liq_quick_ratio"] = ((df["cash_and_equiv"] + df["AR"]) / df["liab_tot"]).where((df["cash_and_equiv"] + df["AR"])!=0, 0)
    df["wc_net_to_asst_tot"] = df["wc_net"] / df["asst_tot"]
    df["asst_tang_fixed_to_asst_tot"] = df["asst_tang_fixed"] / df["asst_tot"]
    df["asst_intang_fixed_to_asst_tot"] = df["asst_intang_fixed"] / df["asst_tot"]
    df["liq_cfo_ratio"] = (df["cf_operations"] / df["liab_tot"]).where(df["cf_operations"]!=0, 0)
    # Profitability Ratio
    df["ebitda_to_asst_tot"] = df["ebitda"] / df["asst_tot"]
    df["prof_operations_to_asst_tot"] = df["prof_operations"]/ df["asst_tot"]
    # Coverage Ratio
    df["ebitda_to_exp_financing"] = (df["ebitda"] / df["exp_financing"]).where(df["ebitda"]!=0, 0)
    # Activity Ratio
    df["AR_to_liab_tot"] = (df["AR"] / df["liab_tot"]).where(df["AR"]!=0, 0) # we don't have current asset or liability
    
    # now handle +/- inf situations, clip with min/max value excluding inf for each column:
    for col in out_cols:
        temp_arr = sorted(df[col].unique())
        df[col] = np.clip(df[col], a_min=temp_arr[1], a_max=temp_arr[-2])
    
    return df

def _convert_dollars_to_asset_ratio(df, cols):
    for col in cols:
        if (col + "_to_asst_tot") not in df.columns:
            df[col + "_to_asst_tot"] = df[col] / df["asst_tot"]
    df = df.drop(cols, axis=1)
    return df
def calc_adj_factor(result, n_q=50):
    """
    Input:
    result : pd.DataFrame() - result should have at least two columns : label / proba (proba : predicted probability from the model) 
    n_q : number of quantiles to use
    
    Output: 
    adjustment : pd.DataFrame() - result would have two columns : bin / adj_factor
    bin : bins used for qcut
    """
    pred_bin, bins = pd.qcut(result["proba"], q=n_q, labels=50 - np.arange(n_q), retbins=True)
    adj_result = result.copy(deep=True)
    adj_result["bin"] = pred_bin
    
    # calc adjustment factor
    actual_prob = adj_result.groupby("bin", as_index=False)["label"].mean()
    pred_prob = adj_result.groupby("bin", as_index=False)["proba"].mean()
    adjustment = actual_prob.merge(pred_prob, on="bin", how="left")
    adjustment["adj_factor"] = adjustment["label"] / adjustment["proba"]
    
    return adjustment[["bin", "adj_factor"]], bins

def apply_adj_factor(result, adj_factor, bins):
    """
    Input:
    result : pd.DataFrame() - result should have at least one column : proba
    adj_factor, bins : output of calc_adj_factor
    
    output
    result : pd.DataFrame() - now proba is adjusted proba
    """
    
    adj_result = result.copy(deep=True)
    n_q = len(bins)-1
    adj_result["bin"] = pd.cut(adj_result["proba"], bins=bins, labels = n_q - np.arange(n_q), include_lowest=True)
    # deal with probability that are outside of the bins (which shouldn't be many..)
    adj_result.loc[(adj_result["bin"].isnull()) & (adj_result["proba"] < bins.min()), "bin"] = n_q
    adj_result.loc[(adj_result["bin"].isnull()) & (adj_result["proba"] > bins.max()), "bin"] = 1
    
    adj_result = adj_result.merge(adj_factor[["bin", "adj_factor"]], on="bin", how="left")
    adj_result["proba"] = adj_result["proba"] * adj_result["adj_factor"]
    
    return adj_result

def preprocessor(df, preproc_params={}, new=True, **kwargs):
    
    #0. Drop unnecessary columns
    for col in df.columns:
        if "Unnamed" in col:
            df = df.drop(col, axis=1)
    if "eqty_corp_family_tot" in df.columns:
        df = df.drop("eqty_corp_family_tot", axis=1)
    
    #1. Date Formats:
    for col in df.columns:
        if "date" in col:
            df[col] = pd.to_datetime(df[col])
    #2. Fill NAs for Default date
    df["def_date"] = df["def_date"].fillna(pd.to_datetime("2099-12-31"))
    
    #3. Generate labels and drop any future-peeking rows
    df["def_min"] = df["stmt_date"] + DateOffset(months=3)
    df["def_max"] = df["stmt_date"] + DateOffset(months=15)
    df["def_in_12mo"] = 0
    df["def_in_12mo"] = df["def_in_12mo"].where(
        (df["def_date"]<df["def_min"]) | (df["def_date"]>df["def_max"]), 1)
    # Future peeking rows
    df = df[~(df["def_date"] < df["def_min"])].drop(["def_min", "def_max"], axis=1)
    
    #4. Merge ATECO industry sectors: csv file should be in data directory:
    df = df.merge(pd.read_csv(data_dir + "ateco_code_industry.csv"), on ="ateco_sector", how="left").drop("ateco_sector", axis=1)  
    # if new, fill na & standardization parameters
    dollar_cols = [col for col in df.columns if col not in ["HQ_city", "INDUSTRY", "def_date", "fs_year", "id", "legal_struct",
                                                            "roa", "roe", "stmt_date", "def_in_12mo", "asst_tot", "days_rec"]]
    num_cols = ["roa", "roe"]
    if new:
        mean_df = pd.DataFrame()
        for col in dollar_cols:
            temp = df.groupby("INDUSTRY", as_index=False).apply(lambda x : (x[col] / x["asst_tot"]).mean())
            temp.columns = ["INDUSTRY", col]
            if mean_df.empty:
                mean_df = temp
            else:
                mean_df = mean_df.merge(temp, on=["INDUSTRY"], how="left")
        for col in num_cols:
            temp = df.groupby("INDUSTRY", as_index=False)[col].mean()
            temp.columns = ["INDUSTRY", col]
            mean_df = mean_df.merge(temp, on=["INDUSTRY"], how="left")
        preproc_params["fill_na_mean"] = mean_df
    
    mean_df = preproc_params["fill_na_mean"]
    
    #5. FILL NA with industry specific means:
    for col in dollar_cols:
        ind_mean = df[["INDUSTRY"]].merge(mean_df[["INDUSTRY", col]], on=["INDUSTRY"], how="left")[col]
        df[col] = df[col].where(~df[col].isnull(), ind_mean * df["asst_tot"])
    for col in num_cols:
        ind_mean = df[["INDUSTRY"]].merge(mean_df[["INDUSTRY", col]], on=["INDUSTRY"], how="left")[col]
        df[col] = df[col].where(~df[col].isnull(), ind_mean)
    
    #6. Calculate all relevant ratios:
    df["liab_tot"] = df["asst_tot"] - df["eqty_tot"]
    dollar_cols.append("liab_tot")
    df = _calc_fin_ratio(df)
    df = _convert_dollars_to_asset_ratio(df, dollar_cols)
    
    #6.1 - drop some factors 
    # This is from factor selection - to reduce multi-colinearity between factors
    factors_to_exclude = ["AR_to_liab_tot", "ebitda_to_asst_tot", "goodwill_to_asst_tot", "liq_quick_ratio", "liq_curr_ratio", 
                      "taxes_to_asst_tot", "AR_to_liab_tot", "prof_operations_to_asst_tot", "liab_tot_to_asst_tot"]
    df = df.drop(factors_to_exclude, axis=1)
    if new:
        preproc_params["train_df"] = df
    
    #7. Calculate year-over-year change variables:
    if not new:
        train_df = preproc_params["train_df"]
        train_df["usage"] = "train"
        df["usage"] = "test"
        df = pd.concat([train_df, df], sort=False, ignore_index=True)
    
    chg_cols = [col for col in df.columns if ((col in ["roe", "roa"]) or ("_to_" in col))]
    df = df.sort_values(["id", "stmt_date"])
    chg_df = df[["id"] + chg_cols].groupby("id").diff().fillna(0)
    chg_df.columns=[c+"_chg_1y" for c in chg_df.columns]
    df = df.join(chg_df)
    
    if not new:
        df = df[df["usage"]=="test"]
        df = df.drop("usage", axis=1)
    
    #8. Merge macro factors:
    m_fac = pd.read_csv(data_dir + "ita_macro_factors.csv")
    df = df.merge(m_fac, on="fs_year", how="left")
    
    
    #8. STANDARDIZE FACTORS
    df["days_rec"] = df.days_rec.fillna(0)
    no_standardize_cols = ['id', 'stmt_date', 'HQ_city', 'legal_struct', 'ateco_sector','def_date', 'fs_year', "def_in_12mo", "INDUSTRY"]
    if new:
        stdz_dict = {}
        for col in df.columns:
            if col not in no_standardize_cols:
                stdz_dict[col] = [df[col].mean(), df[col].std()]
        preproc_params["stdz_dict"] = stdz_dict
    
    stdz_dict = preproc_params["stdz_dict"]
    for col in stdz_dict.keys():
        df[col] = (df[col] - stdz_dict[col][0]) / stdz_dict[col][1]
    
    #9. one-hot encoding for INDUSTRY
    df = df.join(pd.get_dummies(df["INDUSTRY"]))
    df = df.drop(["id", "stmt_date", "INDUSTRY", "fs_year", "legal_struct", "HQ_city", "def_date"], axis=1)
    
    #10. Sort columns to make sure column orders match in the model read back
    df = df[sorted(df.columns)]
    
    
    if new:
        return df, preproc_params
    else:
        if kwargs.get("final", True):
            if "def_in_12mo" in df.columns:
                df = df.drop("def_in_12mo", axis=1)
        return df
    

def estimator(df, fitting_algo, est_params = {}):
    
    model = fitting_algo(est_params)
    
    return(model)

def predictor(new_df, model):
    predictions = model.predict_proba(new_df)[:,1]
    return(predictions)

    
    #10. Sort columns to make sure column orders match in the model read back
    df = df[sorted(df.columns)]
    
    
    if new:
        return df, preproc_params
    else:
        if kwargs.get("final", True):
            if "def_in_12mo" in df.columns:
                df = df.drop("def_in_12mo", axis=1)
        return df

def predictor_harness(new_df, model, preprocessor, preproc_params = {}, **kwargs):
    
    print("started preprocessing test_data")
    proc_df = preprocessor(new_df, preproc_params, new=False)
    print("finished preprocessing test_data - preprocessed shape : %s" % str(proc_df.shape))
    predictions = predictor(proc_df, model)
    
    if kwargs.get("calibration_params", None):
        print("Calibrating Probabililties")
        predictions = pd.DataFrame({"proba" : predictions})
        adj_fact, bins = kwargs["calibration_params"]["adj_fact"], kwargs["calibration_params"]["bins"]
        predictions = apply_adj_factor(predictions, adj_fact, bins)["proba"]
        
    # your code here
    return(predictions)

def main():
    
    parser = optparse.OptionParser()
    
    parser.add_option("-t",
                      "--tr",
                      dest="train_file_name",
                      type="str",
                      default="train_data.csv",
                      help="train csv file name. Needs to be in data directory")
    
    parser.add_option("-v",
                      "--ts",
                      dest="test_file_name",
                      type="str",
                      default="test_data.csv",
                      help="test csv file name. Needs to be in data dictionary")
     
    (options, args) = parser.parse_args()
    
    train_file = options.train_file_name
    test_file = options.test_file_name 
     
    ## We need original train.csv in the data folder as well
    train_data = pd.read_csv(data_dir + train_file)
    print("shape of train_data : %s" % str(train_data.shape))
    # Please rename test_data.csv to match test file name
    test_data = pd.read_csv(data_dir + test_file)
    print("shape of test_data : %s" % str(test_data.shape))
    
    with open(data_dir+"rf_clf_fin.pkl", 'rb') as file:
        rf_clf_read = pickle.load(file)
    
    preproc_train, params = preprocessor(train_data)
    
    adj_factor = pd.read_csv(data_dir+"adj_fact.csv")[["adj_factor", "bin"]]
    bins = pd.read_csv(data_dir+"adj_bin.csv")["bins"]
    calibration_params = {"adj_fact" : adj_factor, "bins": bins}
    
    y_pred = predictor_harness(test_data, rf_clf_read, preprocessor, preproc_params=params, calibration_params=calibration_params)
    
    print("saving output to %s" % data_dir + "team_maroon_output.csv")
    pd.DataFrame(pd.DataFrame({"pd_estimate" : y_pred})).to_csv(data_dir + "team_maroon_output.csv", index=False, header=False)
    print("Done!")

if __name__=="__main__":
    main()


