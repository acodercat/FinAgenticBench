"""Financial ledger, sharp bound certificates and substantive wrong paths."""
import numpy as np
from scipy.optimize import linprog
from evals.calculation_reconciliation import private_fund_waterfall_catchup_clawback as f
from evals.risk_screening import nmfp_maturity_moments_liquidity_bounds as m
from scripts.verdict_baseline import mutate


def test_fund_catchup_trueup_and_escrow():
 capital,fee,times,df=f.inputs();weights=np.array([.4,.35,.25]);investment=(capital-fee)*weights;cash=investment*np.array([1.8,1.4,.4]);lp,gp,ledger=f.waterfall(capital,times,cash)
 assert np.allclose(lp+gp,cash,atol=1e-10,rtol=0)
 # Independent cumulative cash thresholds; capital priority then simple outstanding-capital pref.
 profit3=sum(cash[:2])-capital;pref3=.08*capital*times[0]+.08*(capital-cash[0])*(times[1]-times[0])
 assert gp[1]>0 and gp[1]<.25*pref3
 actual=f.analysis();paid=actual['early_deal_carry_mn'];rho=actual['minimum_early_escrow_pct']/100;claws=[]
 for multiple in [0,.1,.4,1.2]:
  entitlement=max(0,min(.2*(profit3+investment[-1]*multiple),profit3+investment[-1]*multiple-pref3))
  flows=cash.copy();flows[-1]=investment[-1]*multiple
  assert abs(sum(f.waterfall(capital,times,flows)[1])-entitlement)<1e-10
  if multiple in [0,.4,1.2]:claws.append(max(0,paid-entitlement))
 assert paid>sum(gp) and all(rho*paid>=x-1e-10 for x in claws)
 assert (rho-1e-7)*paid<max(claws)
 bad={v.name:x for v,x in zip(f.variables,f.ground_truth())};bad['stress_clawback_mn']=0
 assert not f.validate(bad).success


def independent_worst(R,capped):
 b,B,cash,q=m.inputs();A=np.vstack([np.ones(8),m.RESET,m.FINAL]);h=m.SPREAD;net=q*(1-h);results=[]
 ub=[np.r_[np.zeros(5),np.ones(3)],np.eye(8)[4],np.eye(8)[7]] if capped else None;rhs=[.2,0,0] if capped else None
 for lam in np.r_[0,h/(1-h)]:
  penalty=np.maximum(0,lam*net-q*h)*B
  z=linprog(penalty,A_eq=A,b_eq=b,A_ub=ub,b_ub=rhs,bounds=(0,None),method='highs-ds',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
  assert z.success and np.max(abs(A@z.x-b))<1e-8
  results.append(lam*max(0,R*(B+cash)-cash)-z.fun)
 return max(results)


def test_moment_sharpness_primal_dual_and_capacity():
 b,B,cash,q=m.inputs();pool=B+cash;A=np.vstack([np.ones(8),m.RESET,m.FINAL]);mature=(m.FINAL<=7).astype(float)
 for capped in [False,True]:
  vertices=m.vertices(capped)
  assert np.max(abs(vertices@A.T-b))<1e-8
  assert np.min(vertices)>=-1e-10
  ub=[np.r_[np.zeros(5),np.ones(3)],np.eye(8)[4],np.eye(8)[7]] if capped else None;rhs=[.2,0,0] if capped else None
  for sign in [1,-1]:
   z=linprog(sign*mature,A_eq=A,b_eq=b,A_ub=ub,b_ub=rhs,bounds=(0,None),method='highs')
   assert z.success and abs(z.fun-min(vertices@(sign*mature)))<1e-10
  loss=[]
  for w in vertices:
   cost,nav,sales=m.liquidation(w,.7);loss.append(cost)
   assert np.all(sales>=0) and np.all(sales<=B*w+1e-8)
   assert abs(cash+sales@(q*(1-m.SPREAD))-.7*pool)<1e-8
   assert abs(cost-sales@(q*m.SPREAD))<1e-10
   assert abs(nav*(.3*pool/100)-(cash+B*w@q-.7*pool-cost))<1e-8
  assert abs(max(loss)-independent_worst(.7,capped))<1e-9
 result=m.analysis();budget=.0009*pool
 assert result['worst_70pct_liquidation_cost_mn']>budget>result['capped_70pct_cost_mn']
 boundary=result['robust_redemption_capacity_pct']/100
 assert independent_worst(boundary-1e-6,True)<budget<independent_worst(boundary+1e-6,True)
 assert abs(independent_worst(boundary,True)-budget)<1e-9
 # Legitimate floating reset does not create a seven-day principal maturity.
 reset_short=(m.RESET<=7).astype(float);z=linprog(reset_short,A_eq=A,b_eq=b,bounds=(0,None),method='highs')
 assert z.fun>result['liquidity_lower_pct']/100+.05


def test_strict_legal_rounding_and_each_output_wrong_path():
 for mod in [f,m]:
  truth=dict(zip([v.name for v in mod.variables],mod.ground_truth()));assert mod.validate(truth).success
  rounded={k:(round(v,dp) if dp is not None else v) for (k,v),dp in zip(truth.items(),mod.DECIMALS)};assert mod.validate(rounded).success
  for k in truth:
   bad=truth.copy();bad[k]=mutate(bad[k]);assert not mod.validate(bad).success
  k=next(k for k in truth if isinstance(truth[k],(float,np.floating)));bad=truth.copy();bad[k]=True;assert not mod.validate(bad).success
  for i,names in enumerate(mod.TURN_NAMES,1):assert mod._subset({n:rounded[n] for n in names},names).success
