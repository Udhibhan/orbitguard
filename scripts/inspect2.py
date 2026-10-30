import pandas as pd, numpy as np
ALLOWED = ["event_id","time_to_tca","miss_distance","c_sigma_r","t_sigma_r","risk","max_risk_estimate"]
df = pd.read_csv("data/raw/train_data.csv", usecols=ALLOWED).sort_values(["event_id","time_to_tca"],ascending=[True,False])
def tier(r):
    crit=(r.risk>=-4)|(r.miss_distance<1000); elev=(r.risk>=-6)|(r.miss_distance<=5000)
    return np.where(crit,"Critical",np.where(elev,"Elevated","Safe"))
g=df.groupby("event_id"); last=g.tail(1).set_index("event_id")
has_pre=df[df.time_to_tca>=2].event_id.unique()
esa=last[(last.time_to_tca<=1)&last.index.isin(has_pre)]
for nm,s in [("all train events",last),("ESA filter (>=1 CDM >=2d AND last CDM <=1d)",esa)]:
    t=pd.Series(tier(s)).value_counts(); print(nm,len(s),t.to_dict(), " high-risk(final>=-6):",(s.risk>=-6).sum())
    pc_only=((s.risk>=-4)).sum(); md_only=((s.miss_distance<1000)&(s.risk<-4)).sum(); print("   Critical via Pc:",pc_only," via miss<1km only:",md_only)
sent=6.378136e7
print("sigma sentinel rows t:",(df.t_sigma_r>=sent*0.999).sum()," c:",(df.c_sigma_r>=sent*0.999).sum()," exact value:", df.t_sigma_r.max())
print("events whose final risk==-30 but miss<1km:",((esa.risk<=-30)&(esa.miss_distance<1000)).sum())
print("time_to_tca<0 rows:",(df.time_to_tca<0).sum())
print("risk -30 vs sigma: median c_sigma_r when risk==-30:",df[df.risk<=-30].c_sigma_r.median().round(1)," else:",df[df.risk>-30].c_sigma_r.median().round(1))
print("risk -30 vs miss: median miss when risk==-30:",df[df.risk<=-30].miss_distance.median()," else:",df[df.risk>-30].miss_distance.median())
