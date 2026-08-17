"""nb_pool_calibration — pool both datasets for the FINAL calibration + accuracy numbers.

Single-dataset calibration is unstable at n=18 (lung AUROC=nan with 0 errors; colon
AUROC=0.647 with 1 confident error). Pooling to ~36 clusters gives a meaningful estimate
and directly answers the reviewers' small-n calibration critique.

Reads:  agent_evidence/<DS>/benchmark_10a.json  for DS in the two datasets
Writes: agent_evidence/pooled_calibration.json + printed report

Runner (no DATASET needed):
    exec(open("/lakehouse/default/Files/scRNA/code/nb_pool_calibration.py").read())
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

ROOT = Path("/lakehouse/default/Files/scRNA")/"agent_evidence"
DATASETS = ["GSE131907", "GSE132465"]
LABEL = {"GSE131907":"Lung (LUAD)", "GSE132465":"Colon (CRC)"}

rows = []
per_ds = {}
for ds in DATASETS:
    b = json.load(open(ROOT/ds/"benchmark_10a.json"))
    pc = pd.DataFrame(b["per_cluster"])
    pc["dataset"] = ds
    rows.append(pc)
    per_ds[ds] = dict(n=len(pc), acc=b["overall_accuracy"], sel=b["selective_accuracy"],
                      auroc=b["auroc"], n_abstain=b["n_abstain"],
                      hit1=b["hit_at_1"], hit3=b["hit_at_3"])
df = pd.concat(rows, ignore_index=True)
N = len(df)

# ---- pooled accuracy ----
overall = df["correct"].mean()
sel = df[~df["abstain"]]["correct"].mean()
n_err = int((~df["correct"]).sum())
n_abs = int(df["abstain"].sum())

print("="*70); print("POOLED TWO-DATASET RESULT"); print("="*70)
print(f"{'dataset':14} {'n':>3} {'acc':>6} {'selective':>10} {'abstain':>8} {'Hit@3':>7}")
for ds in DATASETS:
    d = per_ds[ds]
    print(f"{LABEL[ds]:14} {d['n']:>3} {d['acc']:>6.3f} {d['sel']:>10.3f} "
          f"{d['n_abstain']:>8} {d['hit3']:>7.3f}")
print("-"*70)
print(f"{'POOLED':14} {N:>3} {overall:>6.3f} {sel:>10.3f} {n_abs:>8} "
      f"{df['correct'].mean():>7}")  # placeholder Hit filled below
print(f"\nPooled accuracy      : {overall:.3f} ({df.correct.sum()}/{N})")
print(f"Pooled selective acc : {sel:.3f}  (excluding {n_abs} abstention[s])")
print(f"Total errors         : {n_err}   Total abstentions: {n_abs}")

# ---- pooled calibration ----
print("\n" + "="*70); print("POOLED CALIBRATION (n=%d clusters)" % N); print("="*70)
y = df["correct"].astype(int).values
c = df["conf"].values
from sklearn.metrics import roc_auc_score
auroc = roc_auc_score(y, c) if len(set(y)) > 1 else float("nan")
rng = np.random.default_rng(0); boots=[]
for _ in range(5000):
    idx = rng.integers(0, N, N)
    if len(set(y[idx])) > 1:
        boots.append(roc_auc_score(y[idx], c[idx]))
ci = (np.percentile(boots,2.5), np.percentile(boots,97.5)) if boots else (float("nan"),)*2
print(f"Pooled AUROC (confidence -> correctness): {auroc:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
print(f"Mean confidence | correct : {df[df.correct].conf.mean():.3f}")
print(f"Mean confidence | wrong   : {df[~df.correct].conf.mean() if n_err else float('nan'):.3f}")

# reliability table (confidence bins vs empirical accuracy)
print("\nReliability (confidence bin -> empirical accuracy):")
bins = [(0.0,0.7),(0.7,0.85),(0.85,0.95),(0.95,1.01)]
for lo,hi in bins:
    m = (df.conf>=lo)&(df.conf<hi)
    if m.any():
        print(f"  conf [{lo:.2f},{hi:.2f}): n={m.sum():>2}  accuracy={df[m].correct.mean():.3f}")

# threshold sweep pooled
print("\nSelective prediction (pooled):")
print("  threshold | n_kept | accuracy")
for thr in [0.5,0.6,0.7,0.8,0.9,0.95]:
    kept = df[df.conf>=thr]
    if len(kept):
        print(f"     >={thr:.2f}  | {len(kept):>6} | {kept.correct.mean():.3f}")

# ---- all errors, named ----
print("\n" + "="*70); print("ALL ERRORS (pooled)"); print("="*70)
for _, r in df[~df.correct].iterrows():
    print(f"  {LABEL[r['dataset']]:14} cluster {r['cluster']}: "
          f"pred={r['call']} | author={r['author']} | conf={r['conf']:.2f} | "
          f"{'abstained' if r['abstain'] else 'committed'}")

json.dump(dict(pooled_n=int(N), pooled_accuracy=float(overall),
               pooled_selective=float(sel), pooled_auroc=float(auroc),
               auroc_ci=[float(ci[0]),float(ci[1])], n_errors=n_err, n_abstain=n_abs,
               per_dataset={ds:per_ds[ds] for ds in DATASETS}),
          open(ROOT/"pooled_calibration.json","w"), indent=2, default=str)

print("\n" + "="*60 + f"""
POOLING COMPLETE
{"="*60}
Pooled accuracy   : {overall:.3f} ({df.correct.sum()}/{N})
Pooled AUROC      : {auroc:.3f}  CI [{ci[0]:.3f}, {ci[1]:.3f}]
Errors/Abstain    : {n_err} / {n_abs}
Output            : pooled_calibration.json""")
