"""Create analytical figures and evaluation-bootstrap intervals from a run."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, f1_score
from src.simulation.block_bootstrap import moving_block_indices

def generate_core_figures(run_dir: str|Path, raw: pd.DataFrame, seed: int=42, replicates: int=200, block_length: int=40) -> None:
    run=Path(run_dir); plots=run/"plots"; plots.mkdir(exist_ok=True); sns.set_theme(style="whitegrid")
    pred=pd.read_csv(run/"predictions_multihorizon.csv",parse_dates=["date"]); metrics=pd.read_csv(run/"classification_metrics.csv"); probs=pd.read_csv(run/"probability_metrics.csv"); trans=pd.read_csv(run/"hmm_transition_matrix.csv"); mc=pd.read_csv(run/"monte_carlo_summary.csv"); paths=pd.read_csv(run/"monte_carlo_paths_sample.csv")
    fig,ax=plt.subplots(figsize=(13,5)); ax.plot(raw.date,raw.close,color="#183153"); ax.set(title="VN-Index – toàn bộ dữ liệu thực tế",xlabel="Ngày",ylabel="Điểm"); fig.tight_layout(); fig.savefig(plots/"01_vnindex_history.png",dpi=160); plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=False)
    for ax,(h,g) in zip(axes,pred.groupby("horizon")):
        cols=[f"calibrated_prob_{c}" for c in ("bull","sideway","bear","stress")]; ax.stackplot(g.date,*[g[c] for c in cols],labels=["Bull","Sideway","Bear","Stress"],alpha=.8); ax.set(title=f"Xác suất đã calibration – test – horizon {h}",ylabel="Xác suất",ylim=(0,1))
    axes[0].legend(ncol=4,loc="upper center"); axes[-1].set_xlabel("Ngày"); fig.tight_layout(); fig.savefig(plots/"02_calibrated_probability_timelines.png",dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,(h,g) in zip(axes,pred.groupby("horizon")):
        tab=pd.crosstab(g.actual_state,g.predicted_state).reindex(index=["Bull","Sideway","Bear","Stress"],columns=["Bull","Sideway","Bear","Stress"],fill_value=0); sns.heatmap(tab,annot=True,fmt="d",cmap="Blues",ax=ax); ax.set_title(f"Confusion matrix – test – {h} phiên")
    fig.tight_layout(); fig.savefig(plots/"03_confusion_matrices.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6)); sns.barplot(data=metrics,x="horizon",y="balanced_accuracy",hue="model",ax=ax); ax.set(title="Balanced Accuracy theo horizon – test",xlabel="Horizon (phiên)",ylabel="Balanced Accuracy"); fig.tight_layout(); fig.savefig(plots/"04_balanced_accuracy_macd.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6)); sns.barplot(data=probs,x="horizon",y="multiclass_brier",hue="model",ax=ax); ax.set(title="Brier score đa lớp – thấp hơn tốt hơn",xlabel="Horizon (phiên)",ylabel="Brier score"); fig.tight_layout(); fig.savefig(plots/"05_brier_macd.png",dpi=160); plt.close(fig)
    matrix=trans.query("horizon==60").pivot(index="from_state",columns="to_state",values="probability"); fig,ax=plt.subplots(figsize=(6,5)); sns.heatmap(matrix,annot=True,fmt=".2f",cmap="mako",vmin=0,vmax=1,ax=ax); ax.set_title("Filtered HMM transition matrix – horizon 60 context"); fig.tight_layout(); fig.savefig(plots/"06_hmm_transition_matrix.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,6)); x=np.arange(paths.shape[1]-1); sample=paths.drop(columns="path_id").to_numpy();
    for lo,hi,alpha in ((.05,.95,.15),(.10,.90,.22),(.25,.75,.32)): ax.fill_between(x,np.quantile(sample,lo,axis=0),np.quantile(sample,hi,axis=0),alpha=alpha,label=f"{round((hi-lo)*100)}%")
    ax.plot(x,np.median(sample,axis=0),color="black",label="Median"); ax.set(title="Structural Monte Carlo fan chart – latest date – simulated",xlabel="Phiên tương lai",ylabel="VN-Index (điểm)"); ax.legend(); fig.tight_layout(); fig.savefig(plots/"07_monte_carlo_fan_latest.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5)); sns.barplot(data=mc,x="horizon",y="probability_max_drawdown_below_minus_8pct",hue="mode",ax=ax); ax.set(title="Xác suất drawdown dưới -8% – Monte Carlo",xlabel="Horizon",ylabel="Xác suất"); fig.tight_layout(); fig.savefig(plots/"08_monte_carlo_drawdown_risk.png",dpi=160); plt.close(fig)
    # Moving-block evaluation bootstrap on fixed causal test predictions. It is not a model-refit bootstrap.
    rng=np.random.default_rng(seed); rows=[]
    for h,g in pred.groupby("horizon"):
        g=g.reset_index(drop=True)
        for b in range(replicates):
            idx=moving_block_indices(len(g),min(block_length,len(g)),rng); y=g.actual_state.to_numpy()[idx]; yhat=g.predicted_state.to_numpy()[idx]
            rows.append({"horizon":h,"replicate":b,"balanced_accuracy":balanced_accuracy_score(y,yhat),"macro_f1":f1_score(y,yhat,average="macro")})
    boot=pd.DataFrame(rows); boot.to_csv(run/"bootstrap_metric_draws_evaluation_only.csv",index=False); intervals=boot.groupby("horizon").agg({"balanced_accuracy":["mean",lambda x:x.quantile(.025),lambda x:x.quantile(.975)],"macro_f1":["mean",lambda x:x.quantile(.025),lambda x:x.quantile(.975)]}); intervals.columns=["_".join((a,"mean" if b=="mean" else "q025" if "0" in b else "q975")) for a,b in intervals.columns]; intervals.reset_index().assign(scope="fixed-prediction moving-block evaluation bootstrap; no refit").to_csv(run/"bootstrap_metric_intervals.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,5)); sns.violinplot(data=boot,x="horizon",y="macro_f1",ax=ax); ax.set(title="Moving-block evaluation uncertainty – Macro F1",xlabel="Horizon",ylabel="Macro F1"); fig.tight_layout(); fig.savefig(plots/"09_bootstrap_macro_f1.png",dpi=160); plt.close(fig)
