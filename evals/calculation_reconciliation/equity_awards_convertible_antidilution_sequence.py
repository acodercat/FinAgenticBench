"""IAS 33 hypothetical capital structure; source templates are independent."""
from functools import lru_cache
import numpy as np
from cave_agent import Variable
from core.data import load_financial_statements, load_insider_table
from core.validation import turn_validator, validate_ordered_outputs

TURN_NAMES = [
    ["award_row_count", "opening_basic_eps"],
    ["incremental_option_shares_mn", "option_only_diluted_eps"],
    ["included_convertibles", "profitable_diluted_eps", "profitable_diluted_shares_mn"],
    ["loss_included_potential", "loss_continuing_diluted_eps", "loss_total_diluted_eps"],
    ["converted_basic_shares_mn", "converted_basic_total_eps", "converted_continuing_diluted_eps", "converted_potential_set"],
]
variables = [
    Variable("award_row_count",None,"Store the eligible source option-award row count as an integer."),
    Variable("opening_basic_eps",None,"Store the hypothetical opening basic EPS in USD per share, rounded to 4 decimals."),
    Variable("incremental_option_shares_mn",None,"Store incremental shares from the hypothetical options in millions, rounded to 4 decimals."),
    Variable("option_only_diluted_eps",None,"Store option-only diluted EPS in USD per share, rounded to 4 decimals."),
    Variable("included_convertibles",None,"Store included convertible IDs as a comma-separated string in inclusion order using A, B, C, or NONE for an empty set."),
    Variable("profitable_diluted_eps",None,"Store profitable-scenario diluted EPS in USD per share, rounded to 4 decimals."),
    Variable("profitable_diluted_shares_mn",None,"Store profitable-scenario diluted weighted-average shares in millions, rounded to 4 decimals."),
    Variable("loss_included_potential",None,"Store the included potential-instrument set as a comma-separated string in order OPTIONS,A,B,C, or NONE for an empty set."),
    Variable("loss_continuing_diluted_eps",None,"Store the continuing-operations diluted EPS in the loss scenario in USD per share, rounded to 4 decimals."),
    Variable("loss_total_diluted_eps",None,"Store total diluted EPS in the loss scenario in USD per share, rounded to 4 decimals."),
    Variable("converted_basic_shares_mn",None,"Store basic weighted-average shares after actual conversion in millions, rounded to 4 decimals."),
    Variable("converted_basic_total_eps",None,"Store total basic EPS after actual conversion in USD per share, rounded to 4 decimals."),
    Variable("converted_continuing_diluted_eps",None,"Store continuing-operations diluted EPS after actual conversion in USD per share, rounded to 4 decimals."),
    Variable("converted_potential_set",None,"Store included remaining potential instruments as a comma-separated string in order OPTIONS,A,B,C, or NONE for an empty set."),
]
DECIMALS = [0,4,4,4,None,4,4,None,4,4,4,4,4,None]

@lru_cache(maxsize=1)
def inputs():
    f=load_financial_statements()
    r=f.loc[f.accession.astype(str).eq("0000950170-25-019379")]
    if len(r)!=1:raise ValueError("Annual template must be unique")
    r=r.iloc[0]
    sub=load_insider_table("submission")
    d=load_insider_table("deriv_trans").merge(
        sub[["accession_number","issuercik","document_type"]],
        on="accession_number",validate="many_to_one")
    d=d.loc[d.issuercik.astype(str).str.lstrip("0").eq("1091667")
        &d.document_type.astype(str).eq("4")
        &d.trans_date.astype(str).str.startswith("2024")
        &d.trans_code.astype(str).eq("A")
        &d.security_title.astype(str).eq("Stock Options")
        &d.undlyng_sec_title.astype(str).eq("Class A Common Stock")
        &d.conv_exercise_price.gt(0)&d.undlyng_sec_shares.gt(0)].copy()
    if d.duplicated(["accession_number","deriv_trans_sk"]).any():raise ValueError("Duplicate row keys")
    return float(r.NetIncomeLoss_ytd),float(r.WeightedAverageNumberOfSharesOutstandingBasic_ytd),d

@lru_cache(maxsize=1)
def analysis():
    n,s,d=inputs();p=float(d.conv_exercise_price.median())*1.25
    inc=float((d.undlyng_sec_shares*500*(1-d.conv_exercise_price/p)).clip(lower=0).sum())
    # Incremental after-tax earnings and actual conversion terms, USD and shares.
    earnings=np.array([150e6,240e6,180e6]);shares=np.array([30e6,30e6,15e6])
    numerator=n;denominator=s+inc;included=[]
    for i in sorted(range(3),key=lambda i:earnings[i]/shares[i]):
        trial=(numerator+earnings[i])/(denominator+shares[i])
        if trial<numerator/denominator:
            numerator+=earnings[i];denominator+=shares[i];included.append("ABC"[i])
    post=184/365;actshares=s+shares[1]*post;cont=-.2*n+earnings[1]*post;total=.2*n+earnings[1]*post
    return {"award_row_count":len(d),"opening_basic_eps":n/s,
        "incremental_option_shares_mn":inc/1e6,"option_only_diluted_eps":n/(s+inc),
        "included_convertibles":','.join(included) or "NONE",
        "profitable_diluted_eps":numerator/denominator,"profitable_diluted_shares_mn":denominator/1e6,
        "loss_included_potential":"NONE","loss_continuing_diluted_eps":-.2*n/s,
        "loss_total_diluted_eps":.2*n/s,"converted_basic_shares_mn":actshares/1e6,
        "converted_basic_total_eps":total/actshares,"converted_continuing_diluted_eps":cont/actshares,
        "converted_potential_set":"NONE"}

@lru_cache(maxsize=1)
def ground_truth():return tuple(analysis()[v.name] for v in variables)
def _subset(outputs,names):
    by={v.name:v for v in variables};truth=dict(zip(by,ground_truth()));dp=dict(zip(by,DECIMALS))
    return validate_ordered_outputs(outputs,[by[n] for n in names],[truth[n] for n in names],[dp[n] for n in names])
def validate(outputs):return _subset(outputs,[v.name for v in variables])
validators={f"validate_turn_{i}":turn_validator(lambda outputs,names=names:_subset(outputs,names)) for i,names in enumerate(TURN_NAMES,1)}
