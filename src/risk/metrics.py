"""Portfolio loss and return statistics."""
import numpy as np

def _arr(returns):
    x=np.asarray(returns,dtype=float).reshape(-1); x=x[np.isfinite(x)]
    if len(x)<2: raise ValueError("at least two finite returns required")
    return x

def risk_metrics(returns, risk_free_rate=0.0):
    x=_arr(returns); losses=-x; out={}
    for level in (.95,.99):
        q=float(np.quantile(losses,level)); tail=losses[losses>=q]
        key=int(level*100); out[f"var_{key}"]=q; out[f"cvar_{key}"]=float(tail.mean()) if len(tail) else q
        out[f"historical_var_{key}"]=q; out[f"historical_cvar_{key}"]=out[f"cvar_{key}"]
        out[f"parametric_var_{key}"]=float(-(x.mean()+x.std(ddof=1)*(__import__('statistics').NormalDist().inv_cdf(1-level))))
    wealth=np.cumprod(1+x); peak=np.maximum.accumulate(np.r_[1.0,wealth])[1:]
    dd=wealth/peak-1
    downside=np.minimum(x,0)
    out.update(max_drawdown=float(dd.min()),sharpe=float((x.mean()-risk_free_rate/252)/(x.std(ddof=1)+1e-15)*np.sqrt(252)),sortino=float((x.mean()-risk_free_rate/252)/(np.sqrt(np.mean(downside**2))+1e-15)*np.sqrt(252)),calmar=float((wealth[-1]**(252/len(x))-1)/(abs(dd.min())+1e-15)),annualized_return=float(wealth[-1]**(252/len(x))-1),observations=int(len(x)))
    return out
