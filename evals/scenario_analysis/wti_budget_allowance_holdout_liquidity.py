"""Fixed-horizon forecast calibration, held-out loss and liquidity timing.

Wrong conventions measured on frozen data: retuning on evaluation selects +5
USD/bbl instead of -5; symmetric loss also selects -5 but changes loss values. Terminal-only first-half
funding is8.887991228070176 million versus13.514705513784461 path reserve,
binding March. Rounding daily month means to cents changes held-out selected
loss from235.12739077436675 million to235.07 million.
Five discrete policy comparisons are exact; this is not a fitted continuous
model or a real-time trading backtest. Target year determines the split."""
from functools import lru_cache
from decimal import Decimal
import pandas as pd
from cave_agent import Variable
from core.data import load_expansion_table
from core.validation import turn_validator, validate_ordered_outputs

D = lambda x: Decimal(str(x))

variables = [
    Variable('calibration_month_count', None, 'Store the calibration target-month count as an integer.'),
    Variable('evaluation_month_count', None, 'Store the evaluation target-month count as an integer.'),
    Variable('calibration_mean_error_usd_barrel', None, 'Store the mean signed calibration error in USD per barrel as a finite number, rounded to 4 decimals from unrounded inputs.'),
    Variable('selected_allowance_usd_barrel', None, 'Store the chosen allowance in USD per barrel as an integer from {-10, -5, 0, 5, 10}.'),
    Variable('calibration_loss_mn', None, 'Store the selected calibration planning loss in USD millions as a finite number, rounded to 4 decimals from unrounded inputs.'),
    Variable('calibration_runner_up_margin_mn', None, 'Store the loss advantage over the next-best candidate in USD millions as a finite number, rounded to 4 decimals from unrounded inputs.'),
    Variable('held_out_selected_loss_mn', None, 'Store the selected rule evaluation planning loss in USD millions as a finite number, rounded to 4 decimals from unrounded inputs.'),
    Variable('held_out_zero_loss_mn', None, 'Store the zero-allowance evaluation planning loss in USD millions as a finite number, rounded to 4 decimals from unrounded inputs.'),
    Variable('held_out_underbudget_month_count', None, 'Store the number of strictly underbudgeted evaluation months as an integer.'),
    Variable('first_half_reserve_mn', None, 'Store the minimum opening reserve in USD millions as a finite number, rounded to 4 decimals from unrounded inputs.'),
    Variable('first_binding_month', None, 'Store the earliest binding month as a YYYY-MM string, or the exact token NONE if the required reserve is zero.'),
]
DECIMALS = [0, 0, 4, 0, 4, 4, 4, 4, 0, 4, None]
TURN_1_NAMES = ['calibration_month_count', 'evaluation_month_count', 'calibration_mean_error_usd_barrel']
TURN_2_NAMES = ['selected_allowance_usd_barrel', 'calibration_loss_mn', 'calibration_runner_up_margin_mn']
TURN_3_NAMES = ['held_out_selected_loss_mn', 'held_out_zero_loss_mn', 'held_out_underbudget_month_count']
TURN_4_NAMES = ['first_half_reserve_mn', 'first_binding_month']


def forecast_panel():
    forecasts=load_expansion_table('eia_steo_vintages').copy()
    actuals=load_expansion_table('eia_bulk_wti_daily')
    horizon=[(pd.Period(t,'M')-pd.Period(p,'M')).n for p,t in zip(forecasts.publication_month,forecasts.target_month)]
    forecasts=forecasts[(pd.Series(horizon,index=forecasts.index)==3)&forecasts.value_status_in_vintage.eq('forecast')]
    forecasts=forecasts[forecasts.target_month.between('2024-01','2025-12')]
    if forecasts.target_month.duplicated().any():
        raise ValueError('Duplicate three-month forecast')
    actuals=actuals[actuals.series_id.eq('PET.RWTC.D')].copy()
    if actuals.observation_date.duplicated().any():
        raise ValueError('Duplicate daily actual')
    actuals['month']=actuals.observation_date.str[:7]
    actual_by_month={month:sum((D(x) for x in group.wti_spot_price_usd_per_barrel.dropna()),D(0))/group.wti_spot_price_usd_per_barrel.notna().sum()
        for month,group in actuals.groupby('month') if group.wti_spot_price_usd_per_barrel.notna().any()}
    rows=[(str(r.target_month),D(r.wti_spot_average_usd_per_barrel),actual_by_month[r.target_month])
        for r in forecasts.sort_values('target_month').itertuples() if r.target_month in actual_by_month and pd.notna(r.wti_spot_average_usd_per_barrel)]
    if not rows or any(not x.is_finite() or x<=0 for _,f,a in rows for x in [f,a]):
        raise ValueError('Missing or invalid forecast prices')
    return rows

def loss(rows, allowance):
    return sum((3*max(D(0),a-f-allowance)+max(D(0),f+allowance-a) for _,f,a in rows),D(0))

def liquidity_buffer(rows, allowance):
    if not rows:
        raise ValueError('Empty liquidity period')
    cumulative=D(0);required=D(0);binding='NONE'
    for month,f,a in rows:
        cumulative+=a-f-allowance
        if cumulative>required:
            required=cumulative;binding=month
    return required,binding

def analysis():
    rows=forecast_panel()
    train=[r for r in rows if r[0].startswith('2024')]
    test=[r for r in rows if r[0].startswith('2025')]
    if not train or len(test)!=12:
        raise ValueError('Incomplete calibration/evaluation split')
    candidates=[D(k) for k in [-10,-5,0,5,10]]
    scores={k:loss(train,k) for k in candidates}
    ranked=sorted(candidates,key=lambda k:(scores[k],k));chosen=ranked[0]
    half=[r for r in test if r[0]<='2025-06']
    reserve,binding=liquidity_buffer(half,chosen)
    answer=dict(calibration_month_count=len(train),evaluation_month_count=len(test),
        calibration_mean_error_usd_barrel=float(sum((f-a for _,f,a in train),D(0))/len(train)),
        selected_allowance_usd_barrel=int(chosen),calibration_loss_mn=float(scores[chosen]),
        calibration_runner_up_margin_mn=float(scores[ranked[1]]-scores[chosen]),
        held_out_selected_loss_mn=float(loss(test,chosen)),held_out_zero_loss_mn=float(loss(test,D(0))),
        held_out_underbudget_month_count=sum(a>f+chosen for _,f,a in test),
        first_half_reserve_mn=float(reserve),first_binding_month=binding)
    return answer

@lru_cache(maxsize=1)
def ground_truth():
    answer = analysis()
    return tuple(answer[v.name] for v in variables)


def _validate_subset(outputs, names):
    by_name = {v.name: v for v in variables}
    truth = dict(zip(by_name, ground_truth()))
    precision = dict(zip(by_name, DECIMALS))
    return validate_ordered_outputs(outputs, [by_name[n] for n in names],
        [truth[n] for n in names], [precision[n] for n in names])


def validate_turn_1(outputs): return _validate_subset(outputs, TURN_1_NAMES)
def validate_turn_2(outputs): return _validate_subset(outputs, TURN_2_NAMES)
def validate_turn_3(outputs): return _validate_subset(outputs, TURN_3_NAMES)
def validate_turn_4(outputs): return _validate_subset(outputs, TURN_4_NAMES)
def validate(outputs): return _validate_subset(outputs, [v.name for v in variables])

validators = {
    'validate_turn_1': turn_validator(validate_turn_1),
    'validate_turn_2': turn_validator(validate_turn_2),
    'validate_turn_3': turn_validator(validate_turn_3),
    'validate_turn_4': turn_validator(validate_turn_4),
}
