import sys, pandas as pd, numpy as np
ALLOWED = ["event_id","time_to_tca","miss_distance","relative_speed","relative_position_r",
           "relative_position_t","relative_position_n","c_object_type","c_sigma_r","t_sigma_r",
           "max_risk_estimate","F10","AP","risk"]
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
for name in ["train_data","test_data"]:
    df = pd.read_csv(f"data/raw/{name}.csv", usecols=lambda c: c in ALLOWED)
    print(f"\n===== {name}: {len(df):,} rows, {df.event_id.nunique():,} events, cols={list(df.columns)}")
    print(df.describe(percentiles=[.01,.5,.99]).T.round(3))
    print("NaNs:\n", df.isna().sum()[df.isna().sum()>0])
    print("c_object_type:\n", df.c_object_type.value_counts(dropna=False))
    g = df.groupby("event_id")
    print("CDMs/event:", g.size().describe().round(2).to_dict())
    print("time_to_tca: max per event median", g.time_to_tca.max().median().round(2), "min per event median", g.time_to_tca.min().median().round(3))
    print("risk value_counts top:", df.risk.round(3).value_counts().head(5).to_dict())
    print("risk == -30 rows:", (df.risk<=-30).sum(), " risk<=-6:", (df.risk<-6).mean().round(3))
    # sorted by time desc within event -> last CDM = min time_to_tca
    d = df.sort_values(["event_id","time_to_tca"], ascending=[True,False])
    last = d.groupby("event_id").tail(1)
    pre = d[d.time_to_tca>=2].groupby("event_id").tail(1)
    print(f"final-CDM risk: >=-4 (Pc>=1e-4): {(last.risk>=-4).sum()}, [-6,-4): {((last.risk>=-6)&(last.risk<-4)).sum()}, <-6: {(last.risk<-6).sum()}, ==-30: {(last.risk<=-30).sum()}")
    print(f"events with final CDM time_to_tca<=1: {(last.time_to_tca<=1).sum()} / {len(last)}")
    print(f"events with >=1 CDM at time_to_tca>=2: {pre.event_id.nunique()} / {df.event_id.nunique()}")
    print("final miss_distance <1km:", (last.miss_distance<1000).sum(), " 1-5km:", ((last.miss_distance>=1000)&(last.miss_distance<=5000)).sum())
    # unit sanity: |rel pos| vs miss_distance
    norm = np.sqrt(df.relative_position_r**2+df.relative_position_t**2+df.relative_position_n**2)
    print("median |rel_pos|/miss_distance:", (norm/df.miss_distance).median().round(4))
    # risk vs max_risk_estimate ordering
    print("share risk<=max_risk_estimate:", (df.risk<=df.max_risk_estimate+1e-9).mean().round(4))
