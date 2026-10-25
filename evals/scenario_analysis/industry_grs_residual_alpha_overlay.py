"""Quantitative-hard baseline with independently measured model-variant probes.
The exact model conventions are pinned; alternatives are sensitivity tests, not live traps.
All states use unrounded data. Raw certification: tests/test_raw_certification_quant_cases.py rebuilds this case's inputs from the source-native archive and asserts they match the governed runtime tables value for value.
"""
from functools import lru_cache
import numpy as np
from scipy.stats import f
from cave_agent import Variable
from core.data import load_ken_french_table
from core.validation import validate_ordered_outputs, turn_validator

IND=['nodur', 'durbl', 'manuf', 'enrgy', 'hitec', 'telcm', 'shops', 'hlth', 'utils', 'other']

def grs_inputs(start='2018-01-01', excess=True):
    d=load_ken_french_table('daily_industry_returns').merge(load_ken_french_table('daily_factors'),on='date',validate='one_to_one').sort_values('date')
    cols=[i+'_pct' for i in IND]
    assert d[cols+['mkt_rf_pct','smb_pct','hml_pct','rf_pct']].notna().all().all()
    train=d[d.date.between(start,'2022-12-31')]; test=d[d.date.between('2023-01-01','2023-12-31')]
    def panel(x):return x[cols].to_numpy(float)-(x[['rf_pct']].to_numpy(float) if excess else 0),x[['mkt_rf_pct','smb_pct','hml_pct']].to_numpy(float)
    return *panel(train),*panel(test)

def grs_a(start='2018-01-01',excess=True,denom='ml',factor_adjust=True,holdout=True):
    y,z,yh,zh=grs_inputs(start,excess);n,p=y.shape;k=z.shape[1];x=np.c_[np.ones(n),z]
    q,r=np.linalg.qr(x,mode='reduced');coef=np.linalg.solve(r,q.T@y);a=coef[0];err=y-x@coef
    sigma=err.T@err/(n if denom=='ml' else n-k-1);zc=z-z.mean(0);omega=zc.T@zc/n
    quad_a=float(a@np.linalg.solve(sigma,a));sh=float(z.mean(0)@np.linalg.solve(omega,z.mean(0)))
    statistic=(n-p-k)/p*quad_a/(1+sh if factor_adjust else 1)
    w=np.linalg.solve(sigma,a)/(252*quad_a)
    pnl=(yh-zh@coef[1:])@w if holdout else (y-z@coef[1:])@w
    target=float(252*a@w);lag=1/(252*quad_a)
    out={'training_count':n,'hitec_alpha_pp':float(a[4]),'enrgy_alpha_pp':float(a[3]),'residual_covariance_condition':float(np.linalg.cond(sigma)),
         'regression_orthogonality':float(np.max(np.abs(x.T@err))/n),'grs_f_statistic':statistic,'grs_p_value':float(f.sf(statistic,p,n-p-k)),
         'factor_sharpe_squared':sh,'alpha_mahalanobis_squared':quad_a,
         **{'overlay_'+i+'_weight':float(w[j]) for j,i in enumerate(IND)},
         'overlay_training_volatility_pct':float(np.sqrt(252*w@sigma@w)),
         'overlay_kkt_residual':float(max(abs(target-1),np.max(abs(sigma@w-lag*a)))),
         'holdout_count':len(pnl),'holdout_annual_mean_pct':float(pnl.mean()*252),'holdout_annual_volatility_pct':float(pnl.std(ddof=1)*np.sqrt(252)),
         'holdout_mean_t_statistic':float(pnl.mean()/pnl.std(ddof=1)*np.sqrt(len(pnl)))}
    full=np.c_[y,z];cv=np.cov(full,rowvar=False,bias=True);mu=full.mean(0);allsh=float(mu@np.linalg.solve(cv,mu))
    diagnostics={'minimum_residual_eigenvalue':float(np.linalg.eigvalsh(sigma)[0]),'regression_condition':float(np.linalg.cond(x)),
                 'geometry_residual':abs(allsh-sh-quad_a),'alpha_constraint':target,'p_boundary_margin':abs(out['grs_p_value']-.05)}
    return out,diagnostics

TURN_1_NAMES = ['training_count', 'hitec_alpha_pp', 'enrgy_alpha_pp', 'residual_covariance_condition']

TURN_2_NAMES = ['grs_f_statistic', 'grs_p_value', 'factor_sharpe_squared', 'alpha_mahalanobis_squared']

TURN_3_NAMES = ['overlay_nodur_weight', 'overlay_durbl_weight', 'overlay_manuf_weight', 'overlay_enrgy_weight', 'overlay_hitec_weight', 'overlay_telcm_weight', 'overlay_shops_weight', 'overlay_hlth_weight', 'overlay_utils_weight', 'overlay_other_weight', 'overlay_training_volatility_pct']

TURN_4_NAMES = ['holdout_count', 'holdout_annual_mean_pct', 'holdout_annual_volatility_pct', 'holdout_mean_t_statistic']

variables = [
    Variable('training_count', None, 'Store the training observation count as an integer.'),
    Variable('hitec_alpha_pp', None, 'Store the fitted HiTec daily alpha in percentage points, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('enrgy_alpha_pp', None, 'Store the fitted Enrgy daily alpha in percentage points, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('residual_covariance_condition', None, 'Store the residual covariance spectral condition number as a dimensionless number, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('grs_f_statistic', None, 'Store the GRS F statistic as a dimensionless number, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('grs_p_value', None, 'Store the GRS upper-tail p-value as a decimal probability, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('factor_sharpe_squared', None, 'Store the daily squared maximum factor Sharpe ratio as a dimensionless number, rounded to 7 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('alpha_mahalanobis_squared', None, 'Store the squared alpha Mahalanobis distance as a dimensionless number, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('overlay_nodur_weight', None, 'Store the signed nodur industry holding as a dimensionless exposure multiple, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('overlay_durbl_weight', None, 'Store the signed durbl industry holding as a dimensionless exposure multiple, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('overlay_manuf_weight', None, 'Store the signed manuf industry holding as a dimensionless exposure multiple, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('overlay_enrgy_weight', None, 'Store the signed enrgy industry holding as a dimensionless exposure multiple, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('overlay_hitec_weight', None, 'Store the signed hitec industry holding as a dimensionless exposure multiple, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('overlay_telcm_weight', None, 'Store the signed telcm industry holding as a dimensionless exposure multiple, rounded to 6 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('overlay_shops_weight', None, 'Store the signed shops industry holding as a dimensionless exposure multiple, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('overlay_hlth_weight', None, 'Store the signed hlth industry holding as a dimensionless exposure multiple, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('overlay_utils_weight', None, 'Store the signed utils industry holding as a dimensionless exposure multiple, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('overlay_other_weight', None, 'Store the signed other industry holding as a dimensionless exposure multiple, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('overlay_training_volatility_pct', None, 'Store the overlay annualized fitted residual volatility in percent, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('holdout_count', None, 'Store the held-out observation count as an integer.'),
    Variable('holdout_annual_mean_pct', None, 'Store the held-out arithmetic annualized overlay mean in percent, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('holdout_annual_volatility_pct', None, 'Store the held-out sample annualized overlay volatility in percent, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
    Variable('holdout_mean_t_statistic', None, 'Store the ordinary iid zero-mean t statistic as a dimensionless number, rounded to 4 decimals. Compute from the unrounded carried model or source inputs.'),
]
DECIMALS = [0, 4, 4, 4, 4, 4, 7, 4, 4, 4, 4, 4, 4, 6, 4, 4, 4, 4, 4, 0, 4, 4, 4]

@lru_cache(maxsize=1)
def ground_truth():
    result=grs_a()[0]
    return tuple(result[v.name] for v in variables)

def _validate_subset(outputs,names):
    truth=dict(zip((v.name for v in variables),ground_truth()))
    by_name={v.name:v for v in variables}; places=dict(zip(by_name,DECIMALS))
    result=validate_ordered_outputs(outputs,[by_name[n] for n in names],[truth[n] for n in names],[places[n] for n in names])
    for name in names:
        value=outputs.get(name)
        if ('residual' in name and name not in ('residual_covariance_condition',)) or name=='regression_orthogonality':
            if isinstance(value,(int,float,np.number)) and not isinstance(value,(bool,np.bool_)) and np.isfinite(value) and value<0:
                result.success=False; result.message='negative absolute invariant residual'
                if result.variable_results and name in result.variable_results:result.variable_results[name].update(correct=False,message='negative absolute invariant residual')
    return result
def validate_turn_1(outputs):return _validate_subset(outputs,TURN_1_NAMES)
def validate_turn_2(outputs):return _validate_subset(outputs,TURN_2_NAMES)
def validate_turn_3(outputs):return _validate_subset(outputs,TURN_3_NAMES)
def validate_turn_4(outputs):return _validate_subset(outputs,TURN_4_NAMES)
def validate(outputs):return _validate_subset(outputs,[v.name for v in variables])
validators={f"validate_turn_{i}":turn_validator(globals()[f"validate_turn_{i}"]) for i in range(1,5)}
