"""
warranty anomaly detection — full colab pipeline
generates synthetic claim data (gpu-accelerated where possible) and trains
an xgboost model on the T4 gpu. outputs two files you drop on your vps.

usage:
  cell 1: !pip install -q polars xgboost scikit-learn joblib cupy-cuda12x
  cell 2: upload this file
  cell 3: from xgb_full_pipeline import run_pipeline
          run_pipeline()
  cell 4: from google.colab import files
          files.download('warranty_model_v1.json')
          files.download('categorical_mappings.json')
"""

import math
import json
import glob
import numpy as np
import polars as pl
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import average_precision_score

# check if cupy is available — if so, we can do random number generation on the gpu
# which is nice for the big arrays but not a dealbreaker if it's missing
try:
    import cupy as cp
    _USE_GPU_RNG = True
    print("✅ CuPy detected — numeric generation on GPU")
except ImportError:
    _USE_GPU_RNG = False
    print("⚠️  CuPy not found — numeric generation on CPU")


def _gpu_rand(func, *args, size):
    """generate random numbers on gpu if cupy is around, otherwise just use numpy."""
    if _USE_GPU_RNG:
        r = getattr(cp.random, func)(*args, size=size)
        return cp.asnumpy(r)
    return getattr(np.random, func)(*args, size=size)


# ============================================================================
# data generation — builds realistic warranty claims with gpu-accelerated rng
# ============================================================================

def _generate_dates(n):
    start = datetime(2022, 1, 1)
    days_range = (datetime(2024, 12, 31) - start).days
    offsets = _gpu_rand("randint", 0, days_range + 1, size=n)
    cd = np.array([start + timedelta(days=int(d)) for d in offsets])
    ro = np.array([c - timedelta(days=int(d)) for c, d in zip(cd, _gpu_rand("randint", 0, 8, size=n))])
    pd_ = np.array([c - timedelta(days=int(d)) for c, d in zip(cd, _gpu_rand("randint", 30, 1801, size=n))])
    return cd, ro, pd_


def _compute_taxes(df):
    n = len(df)
    pre = (df["Part_Cost"] + df["Labour"] + df["Sublet"]).to_numpy()
    mask = _gpu_rand("random", size=n) < 0.70
    return df.with_columns([
        pl.Series("IGST", np.where(mask, pre * 0.18, 0.0), dtype=pl.Float64),
        pl.Series("CGST", np.where(mask, 0.0, pre * 0.09), dtype=pl.Float64),
        pl.Series("SGST", np.where(mask, 0.0, pre * 0.09), dtype=pl.Float64),
    ])


def _generate_chunk(n, idx):
    rng = np.random.default_rng()
    CT = ["Campaign", "TMA", "Regular", "Free Service Labor Claim"]
    PT = ["NONCS1000PARTS", "RS10000PARTS"]
    CA = ["ZZ2", "ZZ3", "ZZ4", "ZZ7"]
    NA = ["L23","L24","L31","W11","W13","W17","B32","B33","D91","D92","A38","Q26","V84","V88","DA1","DJ6"]
    ST = ["Open", "Pending", "Accept", "Suspense(P)"]
    DL = ["Modi Hyundai", "Viva Honda", "Modi Motors Mumbai", "Modi Motors Pune"]
    PD = ["VCU UNIT","RING SET-PISTON","BRAKE PAD SET","OIL FILTER","AIR FILTER",
          "SPARK PLUG","CLUTCH PLATE","RADIATOR ASSY","ALTERNATOR","STARTER MOTOR",
          "WATER PUMP","TIMING BELT","FUEL INJECTOR","OXYGEN SENSOR","CATALYTIC CONVERTER",
          "SHOCK ABSORBER","STEERING RACK","WHEEL BEARING","CV JOINT","HEADLAMP ASSY"]

    base = idx * n
    chars = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    # GPU-accelerated numerics
    mileage = np.clip(_gpu_rand("gamma", 2, 10000, size=n), 0, 200000).astype(int)
    pc = np.zeros(n, dtype=np.float64)
    mask = _gpu_rand("random", size=n) >= 0.40
    if mask.sum() > 0:
        pc[mask] = _gpu_rand("lognormal", 7, 1, size=int(mask.sum()))
    labour = np.clip(_gpu_rand("normal", 500, 200, size=n), 100, 5000)
    sublet = np.zeros(n, dtype=np.float64)
    sm = _gpu_rand("random", size=n) >= 0.90
    if sm.sum() > 0:
        sublet[sm] = _gpu_rand("uniform", 100, 2000, size=int(sm.sum()))

    cd, ro, pd_ = _generate_dates(n)

    df = pl.DataFrame({
        "S_NO": np.arange(base+1, base+n+1),
        "VIN": ["MAL"+"".join(rng.choice(chars, size=12)) for _ in range(n)],
        "Claim_No": [f"CLM{base+j+1:010d}" for j in range(n)],
        "ACL_No": [f"ACL{rng.integers(100000,999999)}" for _ in range(n)],
        "Claim_Date": cd,
        "Claim_Type": rng.choice(CT, size=n).tolist(),
        "RO_No": [f"RO{rng.integers(100000,999999)}" for _ in range(n)],
        "RO_Date": ro,
        "Status": rng.choice(ST, size=n).tolist(),
        "Mileage": mileage,
        "Cause": rng.choice(CA, size=n).tolist(),
        "Nature": rng.choice(NA, size=n).tolist(),
        "Causal_Part": [f"CP{rng.integers(10000,99999)}" for _ in range(n)],
        "Main_OP": [f"OP{rng.integers(1000,9999)}" for _ in range(n)],
        "Part_Desc": rng.choice(PD, size=n).tolist(),
        "Part_Cost": pc, "Labour": labour, "Sublet": sublet,
        "Invoice_No": [f"INV{rng.integers(100000,999999)}" for _ in range(n)],
        "Part_Type": rng.choice(PT, size=n).tolist(),
        "Pdctn_Date": pd_,
        "Dealership": rng.choice(DL, size=n).tolist(),
    })
    df = _compute_taxes(df)
    df = df.with_columns(
        (pl.col("Part_Cost")+pl.col("Labour")+pl.col("Sublet")+pl.col("IGST")+pl.col("CGST")+pl.col("SGST")).alias("Total_Amt")
    )
    ta = df["Total_Amt"].to_numpy()
    df = df.with_columns(pl.Series("Approve_Amount_by_HMI", ta * _gpu_rand("uniform", 0.90, 1.00, size=n), dtype=pl.Float64))
    return df


def _apply_anomaly_labels(df):
    n = len(df)
    rng = np.random.default_rng()
    for c in ["Claim_Date","RO_Date","Pdctn_Date"]:
        if df.schema[c] == pl.Object:
            df = df.with_columns(pl.Series(c, df[c].to_list(), dtype=pl.Datetime("us")))

    def _detect(f):
        pt = pl.col("Part_Cost")+pl.col("Labour")+pl.col("Sublet")
        tx = pl.col("IGST")+pl.col("CGST")+pl.col("SGST")
        sp = pl.when(pt>1e-6).then(pt).otherwise(1e-6)
        ud = 86400*1_000_000
        f = f.with_columns((
            (pl.col("Part_Cost")>35000)|((pl.col("Mileage")<1000)&(pl.col("Part_Cost")>8000))|
            (((tx-pt*0.18).abs()/sp)>0.01)|
            (((pl.col("Claim_Date")-pl.col("Pdctn_Date")).dt.total_microseconds()>1825*ud)&(pl.col("Claim_Type")=="Regular"))|
            ((pl.col("Approve_Amount_by_HMI")<pl.col("Total_Amt")*0.50)&(pl.col("Status")=="Accept"))
        ).cast(pl.Int8).alias("Is_Anomaly"))
        sf = f.sort(["VIN","Claim_Date"])
        vs = sf["VIN"].to_list(); ds = sf["Claim_Date"].to_list(); ar = sf["Is_Anomaly"].to_numpy().copy()
        i = 0
        while i < len(vs):
            j = i
            while j < len(vs) and vs[j]==vs[i]: j+=1
            if j-i>3:
                gd = ds[i:j]
                for k in range(len(gd)):
                    cnt = sum(1 for m in range(len(gd)) if abs((gd[k]-gd[m]).total_seconds())/86400<=30)
                    if cnt>3:
                        for m in range(len(gd)):
                            if abs((gd[k]-gd[m]).total_seconds())/86400<=30: ar[i+m]=1
            i = j
        return sf.with_columns(pl.Series("Is_Anomaly", ar, dtype=pl.Int8))

    df = _detect(df)
    nat = int(df["Is_Anomaly"].sum())
    if nat/n < 0.003:
        need = int(n*0.005)-nat
        if need > 0:
            cl = df.filter(pl.col("Is_Anomaly")==0).with_row_index("_idx")["_idx"].to_list()
            ch = rng.choice(cl, size=min(need,len(cl)), replace=False)
            ni = len(ch); sp = [int(ni*0.25),int(ni*0.20),int(ni*0.20),int(ni*0.20)]; sp.append(ni-sum(sp))
            pc=df["Part_Cost"].to_numpy().copy(); mi=df["Mileage"].to_numpy().copy()
            vl=df["VIN"].to_list(); cdl=list(df["Claim_Date"].to_list())
            ig=df["IGST"].to_numpy().copy(); cg=df["CGST"].to_numpy().copy(); sg=df["SGST"].to_numpy().copy()
            pdl=list(df["Pdctn_Date"].to_list()); ctl=df["Claim_Type"].to_list()
            ap=df["Approve_Amount_by_HMI"].to_numpy().copy(); stl=df["Status"].to_list()
            ta=df["Total_Amt"].to_numpy().copy(); la=df["Labour"].to_numpy().copy(); su=df["Sublet"].to_numpy().copy()
            c0=ch[:sp[0]]; c1=ch[sp[0]:sp[0]+sp[1]]; c2=ch[sp[0]+sp[1]:sp[0]+sp[1]+sp[2]]
            c3=ch[sp[0]+sp[1]+sp[2]:sp[0]+sp[1]+sp[2]+sp[3]]; c4=ch[sp[0]+sp[1]+sp[2]+sp[3]:]
            for x in c0:
                if rng.random()<0.5: pc[x]=rng.uniform(36000,60000)
                else: mi[x]=rng.integers(0,999); pc[x]=rng.uniform(8500,15000)
            for x in c1:
                p=pc[x]+la[x]+su[x]; ig[x]=p*(0.18+rng.choice([-1,1])*rng.uniform(0.03,0.10)); cg[x]=0; sg[x]=0
            for x in c2:
                pdl[x]=cdl[x]-timedelta(days=int(rng.integers(1826,2500))); ctl[x]="Regular"
            for c in range(max(1,len(c3)//5)):
                ci=c3[c*5:min((c+1)*5,len(c3))]
                if len(ci)<4:
                    for x in ci: pc[x]=rng.uniform(36000,60000)
                    continue
                sv="MAL"+"".join(rng.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),size=12))
                bd=cdl[ci[0]]
                for x in ci: vl[x]=sv; cdl[x]=bd+timedelta(days=int(rng.integers(0,16)))
            for x in c4: ap[x]=ta[x]*rng.uniform(0.10,0.45); stl[x]="Accept"
            for x in c0:
                p=pc[x]+la[x]+su[x]; ig[x]=p*0.18; cg[x]=0; sg[x]=0; ta[x]=p+ig[x]; ap[x]=ta[x]*rng.uniform(0.90,1.00)
            for x in c1: ta[x]=pc[x]+la[x]+su[x]+ig[x]
            df = df.with_columns([
                pl.Series("Part_Cost",pc,dtype=pl.Float64),pl.Series("Mileage",mi),pl.Series("VIN",vl),
                pl.Series("Claim_Date",cdl,dtype=pl.Datetime("us")),pl.Series("IGST",ig,dtype=pl.Float64),
                pl.Series("CGST",cg,dtype=pl.Float64),pl.Series("SGST",sg,dtype=pl.Float64),
                pl.Series("Pdctn_Date",pdl,dtype=pl.Datetime("us")),pl.Series("Claim_Type",ctl),
                pl.Series("Approve_Amount_by_HMI",ap,dtype=pl.Float64),pl.Series("Status",stl),
                pl.Series("Total_Amt",ta,dtype=pl.Float64),
            ])
            df = df.drop("Is_Anomaly"); df = _detect(df)
    cur = int(df["Is_Anomaly"].sum())
    if cur/len(df)>0.01:
        mx=int(len(df)*0.01); exc=cur-mx
        if exc>0:
            ai=df.with_row_index("_idx").filter(pl.col("Is_Anomaly")==1)["_idx"].to_list()
            fl=set(rng.choice(ai,size=exc,replace=False).tolist()); a=df["Is_Anomaly"].to_numpy().copy()
            for x in fl: a[x]=0
            df=df.with_columns(pl.Series("Is_Anomaly",a,dtype=pl.Int8))
    return df


def generate_data(total_records=10_000_000, chunk_size=1_000_000):
    total_chunks = math.ceil(total_records / chunk_size)
    for i in range(total_chunks):
        sz = min(chunk_size, total_records - i*chunk_size)
        chunk = _generate_chunk(sz, i)
        chunk = _apply_anomaly_labels(chunk)
        chunk.write_parquet(f"claims_batch_{i}.parquet")
        print(f"✅ Chunk {i+1}/{total_chunks} generated ({sz:,} rows)")
    print(f"\n✅ All {total_records:,} rows generated")


# ============================================================================
# training — xgboost with cuda gpu acceleration
# ============================================================================

# same column lists as the local trainer, just defined here so this file is self-contained
CATEGORICAL_COLS = ["Claim_Type", "Part_Type", "Cause", "Nature", "Status", "Dealership"]
FEATURE_COLS = [
    "Mileage","Part_Cost","Labour","Sublet","Total_Amt","IGST","CGST","SGST",
    "Approve_Amount_by_HMI","Vehicle_Age_Days","Claim_RO_Gap_Days","Tax_Rate","Approval_Ratio",
    "Claim_Type_idx","Part_Type_idx","Cause_idx","Nature_idx","Status_idx","Dealership_idx",
]


def _encode_categoricals(df):
    mappings = {}
    for col in CATEGORICAL_COLS:
        uv = sorted(df[col].unique().to_list())
        m = {v: i for i, v in enumerate(uv)}
        mappings[col] = m
        df = df.with_columns(pl.col(col).replace_strict(m).cast(pl.Int64).alias(col+"_idx"))
    with open("categorical_mappings.json","w") as f: json.dump(mappings, f, indent=2)
    print("✅ Categorical mappings saved")
    return df


def _engineer_features(df):
    ud = 86400*1_000_000
    df = df.with_columns([
        ((pl.col("Claim_Date")-pl.col("Pdctn_Date")).dt.total_microseconds()/ud).alias("Vehicle_Age_Days"),
        ((pl.col("Claim_Date")-pl.col("RO_Date")).dt.total_microseconds()/ud).alias("Claim_RO_Gap_Days"),
    ])
    pt = pl.col("Part_Cost")+pl.col("Labour")+pl.col("Sublet")
    sp = pl.when(pt>1e-6).then(pt).otherwise(pl.lit(1e-6))
    df = df.with_columns(((pl.col("IGST")+pl.col("CGST")+pl.col("SGST"))/sp).alias("Tax_Rate"))
    st = pl.when(pl.col("Total_Amt")>1e-6).then(pl.col("Total_Amt")).otherwise(pl.lit(1e-6))
    df = df.with_columns((pl.col("Approve_Amount_by_HMI")/st).alias("Approval_Ratio"))
    return df


def train_model(data_glob="claims_batch_*.parquet"):
    files_list = sorted(glob.glob(data_glob))
    if not files_list: raise FileNotFoundError(f"No files matching '{data_glob}'")
    print(f"📂 Loading {len(files_list)} files...")
    df = pl.scan_parquet(data_glob).collect()
    print(f"   {len(df):,} rows")
    df = _encode_categoricals(df)
    df = _engineer_features(df)
    X = df.select(FEATURE_COLS).to_pandas()
    y = df.select("Is_Anomaly").to_pandas().values.ravel()
    pos=int(y.sum()); neg=len(y)-pos
    if pos==0 or neg==0: raise ValueError(f"No variation: pos={pos}, neg={neg}")
    print(f"📊 {pos:,} anomalies ({pos/len(y)*100:.2f}%), {neg:,} normal")
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    spw = (len(y_train)-int(y_train.sum()))/int(y_train.sum())
    print(f"⚖️  scale_pos_weight: {spw:.2f}")

    use_gpu = True
    try:
        xgb.train({"tree_method":"hist","device":"cuda","verbosity":0,"max_depth":2},
                   xgb.DMatrix(X_train.iloc[:100],label=y_train[:100]),num_boost_round=1)
        print("🚀 GPU confirmed — training on CUDA T4")
    except Exception as e:
        print(f"⚠️  GPU test failed ({e}), using CPU"); use_gpu=False

    params = {
        "objective":"binary:logistic","eval_metric":"aucpr","scale_pos_weight":spw,
        "learning_rate":0.05,"max_depth":6,"min_child_weight":50,
        "subsample":0.8,"colsample_bytree":0.8,
        "tree_method":"hist",
        "device":"cuda" if use_gpu else "cpu",
        "verbosity":1,"nthread":-1,
    }

    cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    scores = []
    print("\n📊 5-Fold CV:")
    for fold,(ti,vi) in enumerate(cv.split(X_train,y_train)):
        m = xgb.train(params,xgb.DMatrix(X_train.iloc[ti],label=y_train[ti]),num_boost_round=1000,
                       evals=[(xgb.DMatrix(X_train.iloc[vi],label=y_train[vi]),"val")],
                       early_stopping_rounds=50,verbose_eval=100)
        p = m.predict(xgb.DMatrix(X_train.iloc[vi]))
        s = average_precision_score(y_train[vi],p); scores.append(s)
        print(f"  Fold {fold+1}/5 PR-AUC: {s:.4f}")
    print(f"📊 Mean: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    X_tr,X_val,y_tr,y_val = train_test_split(X_train,y_train,test_size=0.1,stratify=y_train,random_state=42)
    print("\n🚀 Final training...")
    model = xgb.train(params,xgb.DMatrix(X_tr,label=y_tr),num_boost_round=1000,
                       evals=[(xgb.DMatrix(X_val,label=y_val),"val")],
                       early_stopping_rounds=50,verbose_eval=100)

    imp = model.get_score(importance_type="gain")
    idf = pd.DataFrame([{"feature":k,"gain":v} for k,v in imp.items()]).sort_values("gain",ascending=False)
    mx = idf["gain"].max() if len(idf)>0 else 1
    print("\n📊 Feature Importance:")
    print("="*60)
    for _,r in idf.iterrows():
        print(f"  {r['feature']:25s} {r['gain']:12.1f}  {'█'*int(r['gain']/mx*30)}")

    pr = average_precision_score(y_test, model.predict(xgb.DMatrix(X_test)))
    print(f"\n🎯 Test PR-AUC: {pr:.4f}")
    model.save_model("warranty_model_v1.json")
    print("💾 warranty_model_v1.json saved")
    print("💾 categorical_mappings.json saved")
    print("\n✅ Download warranty_model_v1.json + categorical_mappings.json for VPS")


# ============================================================================
# run everything
# ============================================================================

def run_pipeline(total_records=10_000_000):
    """one-shot: generate data then train. call this from a colab cell."""
    print("=" * 60)
    print("PHASE 1: Data Generation")
    print("=" * 60)
    generate_data(total_records=total_records)

    print("\n" + "=" * 60)
    print("PHASE 2: XGBoost GPU Training")
    print("=" * 60)
    train_model()


if __name__ == "__main__":
    run_pipeline()
