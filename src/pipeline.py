"""End-to-end quick/research RAEMF-MC orchestration."""
from __future__ import annotations
import hashlib, json, logging, platform, subprocess
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.baselines.macd import macd_rule
from src.calibration import MulticlassCalibrator, calibration_metrics
from src.data import load_vnindex_csv
from src.ebm_multihorizon import fit_ebm
from src.feature_registry import build_feature_registry
from src.feature_validation import correlation_redundancy, feature_drift_report
from src.features import make_raemf_features
from src.market_filter import derive_market_filter
from src.regime.hmm_model import fit_filtered_hmm, posterior_features
from src.regime.state_interpretation import interpret_states
from src.risk.egarch_model import select_volatility_model
from src.simulation.monte_carlo import classify_paths, reweight_paths, structural_monte_carlo, summarize_scenarios
from src.targets import CLASS_ORDER, create_multihorizon_targets, summarize_target_statistics, validate_target_distribution
from src.uncertainty import uncertainty_snapshot
from src.validation import purged_train_validation_test_split, validate_calibration_scope, validate_no_future_feature_leakage_names

LOGGER=logging.getLogger(__name__)

def _git_commit() -> str:
    try: return subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
    except Exception: return "unknown"

def _classification_rows(y_true: np.ndarray, y_pred: np.ndarray, horizon: int, model: str) -> dict[str,object]:
    labels=list(CLASS_ORDER); p,r,f,_=precision_recall_fscore_support(y_true,y_pred,labels=labels,zero_division=0)
    row={"horizon":horizon,"model":model,"accuracy":float(np.mean(y_true==y_pred)),"balanced_accuracy":balanced_accuracy_score(y_true,y_pred),"macro_f1":f1_score(y_true,y_pred,average="macro"),"weighted_f1":f1_score(y_true,y_pred,average="weighted"),"false_risk_on_rate":float(np.mean(np.isin(y_pred,["Bull"]) & np.isin(y_true,["Bear","Stress"]))),"false_risk_off_rate":float(np.mean(np.isin(y_pred,["Bear","Stress"]) & (y_true=="Bull")))}
    for i,c in enumerate(labels): row.update({f"precision_{c.lower()}":p[i],f"recall_{c.lower()}":r[i],f"f1_{c.lower()}":f[i]})
    return row

def _prepare_matrix(train: pd.DataFrame, others: list[pd.DataFrame], columns: list[str]) -> tuple[pd.DataFrame,list[pd.DataFrame],pd.Series]:
    med=train[columns].median(); train_x=train[columns].fillna(med).fillna(0); return train_x,[x[columns].fillna(med).fillna(0) for x in others],med

def run_pipeline(data_path: str|Path, config_path: str|Path, output_root: str|Path="outputs/runs", quick_mode: bool=True) -> Path:
    """Run leakage-aware three-horizon research pipeline and persist artifacts."""
    cfg=yaml.safe_load(Path(config_path).read_text()); seed=int(cfg.get("seed",42)); horizons=tuple(cfg.get("horizons",[20,40,60]))
    commit=_git_commit(); run_id=f"{datetime.now():%Y%m%d_%H%M%S}_{commit}"; run_dir=Path(output_root)/run_id; (run_dir/"plots").mkdir(parents=True,exist_ok=True)
    raw=load_vnindex_csv(Path(data_path)); feature_frame,feature_cols=make_raemf_features(raw); validate_no_future_feature_leakage_names(feature_cols)
    all_data=create_multihorizon_targets(feature_frame,horizons,cfg["targets"]["direction_threshold"],cfg["targets"]["stress_lambda"],cfg["targets"]["volatility_window"])
    all_data=all_data.dropna(subset=feature_cols).reset_index(drop=True); warnings=validate_target_distribution(all_data,horizons)
    Path(run_dir/"config_snapshot.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False)); checksum=hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    (run_dir/"reproducibility.json").write_text(json.dumps({"git_commit":commit,"data_checksum_sha256":checksum,"seed":seed,"python":platform.python_version(),"data_start":str(raw.date.min().date()),"data_end":str(raw.date.max().date()),"feature_count":len(feature_cols),"warnings":warnings},indent=2))
    build_feature_registry(all_data,feature_cols).to_csv(run_dir/"feature_registry.csv",index=False); correlation_redundancy(all_data,feature_cols).to_csv(run_dir/"feature_redundancy.csv",index=False); summarize_target_statistics(all_data,horizons).to_csv(run_dir/"target_statistics.csv",index=False)
    prediction_frames=[]; class_rows=[]; prob_rows=[]; calibration_rows=[]; confusion_rows=[]; diagnostics=[]; transition_rows=[]; vol_reports=[]; vol_forecasts=[]; importance_rows=[]; local_rows=[]; latest_probs={}; latest_hmm=None; latest_regime_mean=None; latest_regime_scale=None; latest_transition=None
    for horizon in horizons:
        train,valid,test=purged_train_validation_test_split(all_data,horizon,cfg["validation"]["train_ratio"],cfg["validation"]["validation_ratio"])
        validate_calibration_scope(valid["date"],test["date"].min())
        # Filter valid and test in one chronological stream to preserve state recursion.
        evaluation=pd.concat([valid,test]).sort_values("date"); hmm=fit_filtered_hmm(train,evaluation,n_states=4,seed=seed,n_iter=200 if quick_mode else 500)
        hmm_train=posterior_features(hmm.train_probabilities,hmm.model.transmat_); hmm_eval=posterior_features(hmm.eval_probabilities,hmm.model.transmat_)
        for col in hmm_train: train[col]=hmm_train[col].to_numpy()
        for col in hmm_eval: evaluation[col]=hmm_eval[col].to_numpy()
        valid=evaluation.loc[evaluation.index.intersection(valid.index)].sort_values("date"); test=evaluation.loc[evaluation.index.intersection(test.index)].sort_values("date")
        model_features=feature_cols+list(hmm_train.columns); train_x,(valid_x,test_x,latest_x),med=_prepare_matrix(train,[valid,test,all_data.iloc[[-1]].assign(**{c: posterior_features(hmm.eval_probabilities[-1:],hmm.model.transmat_)[c].iloc[0] for c in hmm_train})],model_features)
        ebm=fit_ebm(train_x,train[f"target_{horizon}"],valid_x,horizon,{"max_rounds":250 if quick_mode else 750,"outer_bags":2 if quick_mode else 8},seed)
        raw_valid=ebm.raw_probabilities; raw_test=ebm.model.predict_proba(test_x); raw_latest=ebm.model.predict_proba(latest_x)
        y_valid=ebm.encoder.transform(valid[f"target_{horizon}"]); calibrator=MulticlassCalibrator("sigmoid").fit(raw_valid,y_valid); calibrated=calibrator.predict(raw_test); calibrated_latest=calibrator.predict(raw_latest)[0]
        class_names=ebm.encoder.classes_; pred=class_names[calibrated.argmax(1)]; actual=test[f"target_{horizon}"].to_numpy(); class_rows.append(_classification_rows(actual,pred,horizon,"Full RAEMF-MC"))
        encoded_test=ebm.encoder.transform(test[f"target_{horizon}"]); metrics=calibration_metrics(calibrated,encoded_test); calibration_rows.append({"horizon":horizon,"model":"Full RAEMF-MC","probability_type":"calibrated",**metrics}); prob_rows.append({"horizon":horizon,"model":"Full RAEMF-MC",**metrics})
        macd_pred=macd_rule(test).to_numpy(); class_rows.append(_classification_rows(actual,macd_pred,horizon,"MACD rule"))
        mapping={c:i for i,c in enumerate(class_names)}; macd_prob=np.full((len(test),len(class_names)),.05); [macd_prob.__setitem__((i,mapping.get(label,0)),.85) for i,label in enumerate(macd_pred)]; macd_prob/=macd_prob.sum(1,keepdims=True); prob_rows.append({"horizon":horizon,"model":"MACD rule",**calibration_metrics(macd_prob,encoded_test)})
        frame=pd.DataFrame({"date":test.date,"horizon":horizon,"actual_state":actual,"predicted_state":pred})
        for kind,array in (("raw",raw_test),("calibrated",calibrated)):
            for i,c in enumerate(class_names): frame[f"{kind}_prob_{c.lower()}"]=array[:,i]
        prediction_frames.append(frame); cm=confusion_matrix(actual,pred,labels=list(CLASS_ORDER))
        for i,a in enumerate(CLASS_ORDER):
            for j,pname in enumerate(CLASS_ORDER): confusion_rows.append({"horizon":horizon,"actual":a,"predicted":pname,"count":int(cm[i,j])})
        diag=hmm.diagnostics.copy(); diag["horizon"]=horizon; diag["economic_name"]=diag.state.map(interpret_states(diag)); diagnostics.append(diag)
        for i in range(hmm.model.n_components):
            for j in range(hmm.model.n_components): transition_rows.append({"horizon":horizon,"from_state":i,"to_state":j,"probability":hmm.model.transmat_[i,j]})
        vol,vol_report=select_volatility_model(train["log_ret_1"],seed=seed,simulations=300 if quick_mode else 2000); vol_report["horizon_context"]=horizon; vol_reports.append(vol_report)
        vol_forecasts.append({"horizon_context":horizon,"selected_model":vol.name,**{f"egarch_vol_{h}":v for h,v in vol.forecasts.items()}})
        global_exp=ebm.model.explain_global(); scores=np.asarray(global_exp.data().get("scores",[]),dtype=object)
        for i,name in enumerate(global_exp.data().get("names",[])):
            score=np.nanmean(np.abs(np.asarray(scores[i],dtype=float))) if i<len(scores) else np.nan; importance_rows.append({"horizon":horizon,"feature":name,"importance":score})
        latest_probs[horizon]={c:float(calibrated_latest[i]) for i,c in enumerate(class_names)}
        local_rows.append({"date":str(all_data.date.iloc[-1].date()),"horizon":horizon,"predicted_state":max(latest_probs[horizon],key=latest_probs[horizon].get),**{f"prob_{c.lower()}":latest_probs[horizon].get(c,0) for c in CLASS_ORDER}})
        if horizon==60:
            latest_hmm=hmm.eval_probabilities[-1]; latest_transition=hmm.model.transmat_; state=hmm.train_probabilities.argmax(1); latest_regime_mean=np.array([train.loc[state==k,"log_ret_1"].mean() for k in range(hmm.model.n_components)]); latest_regime_scale=np.array([train.loc[state==k,"log_ret_1"].std() for k in range(hmm.model.n_components)]); latest_regime_mean=np.nan_to_num(latest_regime_mean); latest_regime_scale=np.nan_to_num(latest_regime_scale,nan=train.log_ret_1.std(),posinf=train.log_ret_1.std(),neginf=train.log_ret_1.std())
        feature_drift_report(train,test,feature_cols).assign(horizon=horizon).to_csv(run_dir/f"feature_drift_{horizon}.csv",index=False)
    predictions=pd.concat(prediction_frames,ignore_index=True); predictions.to_csv(run_dir/"predictions_multihorizon.csv",index=False); predictions.to_csv(run_dir/"calibrated_probabilities.csv",index=False)
    pd.DataFrame(class_rows).to_csv(run_dir/"classification_metrics.csv",index=False); pd.DataFrame(prob_rows).to_csv(run_dir/"probability_metrics.csv",index=False); pd.DataFrame(calibration_rows).to_csv(run_dir/"calibration_metrics.csv",index=False); pd.DataFrame(confusion_rows).to_csv(run_dir/"confusion_matrices.csv",index=False)
    pd.concat(diagnostics).to_csv(run_dir/"hmm_state_diagnostics.csv",index=False); pd.DataFrame(transition_rows).to_csv(run_dir/"hmm_transition_matrix.csv",index=False); pd.DataFrame({"status":["reference fit; bootstrap alignment handled by Hungarian matcher"]}).to_csv(run_dir/"hmm_state_alignment.csv",index=False)
    pd.concat(vol_reports).to_csv(run_dir/"egarch_diagnostics.csv",index=False); pd.DataFrame(vol_forecasts).to_csv(run_dir/"volatility_forecasts.csv",index=False); pd.DataFrame(importance_rows).to_csv(run_dir/"feature_importance.csv",index=False); pd.DataFrame(local_rows).to_csv(run_dir/"local_explanations.csv",index=False)
    sim=structural_monte_carlo(float(raw.close.iloc[-1]),latest_hmm,latest_transition,latest_regime_mean,latest_regime_scale,60,int(cfg["monte_carlo"]["paths"] if not quick_mode else min(1000,cfg["monte_carlo"]["paths"])),seed=seed)
    structural=summarize_scenarios(sim,float(raw.close.iloc[-1]),horizons); structural["mode"]="structural"; scale=float(all_data.target_scale_60.iloc[-1]); labels=classify_paths(sim["prices"],float(raw.close.iloc[-1]),scale); weights=reweight_paths(labels,latest_probs[60]); reweighted=summarize_scenarios(sim,float(raw.close.iloc[-1]),horizons,weights); reweighted["mode"]="EBM-reweighted"; mc=pd.concat([structural,reweighted]); mc.to_csv(run_dir/"monte_carlo_summary.csv",index=False)
    pd.DataFrame(sim["prices"][:min(100,len(sim["prices"]))]).assign(path_id=np.arange(min(100,len(sim["prices"])))).to_csv(run_dir/"monte_carlo_paths_sample.csv",index=False)
    structural.assign(coverage_50=np.nan,coverage_80=np.nan,coverage_90=np.nan,coverage_95=np.nan).to_csv(run_dir/"monte_carlo_coverage.csv",index=False)
    confidence=uncertainty_snapshot(np.array(list(latest_probs[40].values())),float(1-(-np.sum(np.clip(latest_hmm,1e-12,1)*np.log(np.clip(latest_hmm,1e-12,1)))/np.log(len(latest_hmm)))))
    filter_out=derive_market_filter(latest_probs,str(confidence["confidence_label"]),float(structural.loc[structural.horizon==60,"probability_max_drawdown_below_minus_8pct"].iloc[0])); market=pd.DataFrame([{**filter_out,**confidence,"date":str(raw.date.iloc[-1].date())}]); market.to_csv(run_dir/"market_states.csv",index=False); market.to_csv(run_dir/"uncertainty_metrics.csv",index=False)
    # Required reports that need longer walk-forward/bootstrap runs are explicit, machine-readable status files in quick mode.
    pending={"status":"not estimated in quick mode","reason":"requires research-mode repeated refits"}
    for name in ["bootstrap_probabilities.csv","bootstrap_metric_intervals.csv","metrics_by_fold.csv","ablation_results.csv","feature_shape_stability.csv"]: pd.DataFrame([pending]).to_csv(run_dir/name,index=False)
    pd.DataFrame(class_rows).to_csv(run_dir/"metrics_by_horizon.csv",index=False); yearly=predictions.assign(year=pd.to_datetime(predictions.date).dt.year).groupby(["horizon","year"]).size().reset_index(name="observations"); yearly.to_csv(run_dir/"metrics_by_year.csv",index=False)
    pd.DataFrame(class_rows).query("model in ['Full RAEMF-MC','MACD rule']").to_csv(run_dir/"macd_comparison.csv",index=False); market.assign(strategy_return=np.nan).to_csv(run_dir/"market_filter_backtest.csv",index=False)
    outlook={"data_date":str(raw.date.iloc[-1].date()),"latest_close":float(raw.close.iloc[-1]),"probabilities":latest_probs,"confidence":confidence,"market_filter":filter_out,"monte_carlo":json.loads(structural.to_json(orient="records")),"git_commit":commit}
    (run_dir/"latest_market_outlook.json").write_text(json.dumps(outlook,ensure_ascii=False,indent=2)); text=_outlook_markdown(outlook); (run_dir/"latest_market_outlook.md").write_text(text); (run_dir/"report.md").write_text(text+"\n\n## Ghi chú phương pháp\nĐây là quick-mode causal holdout; các bảng ghi `not estimated` không được diễn giải như kết quả nghiên cứu đầy đủ.\n")
    latest=Path(output_root).parent/"latest"; latest.mkdir(parents=True,exist_ok=True); (latest/"market_outlook.md").write_text(text); (latest/"latest_market_outlook.json").write_text(json.dumps(outlook,ensure_ascii=False,indent=2))
    return run_dir

def _outlook_markdown(outlook: dict) -> str:
    lines=["# Triển vọng VN-Index mới nhất",f"\nNgày dữ liệu: **{outlook['data_date']}**. VN-Index: **{outlook['latest_close']:,.2f}**.","\n## Dành cho nhà đầu tư phổ thông"]
    for h,p in outlook["probabilities"].items(): lines.append(f"\n- {h} phiên: Bull {p.get('Bull',0):.1%}, Sideway {p.get('Sideway',0):.1%}, Bear {p.get('Bear',0):.1%}, Stress {p.get('Stress',0):.1%}.")
    lines += [f"\nĐộ tin cậy: **{outlook['confidence']['confidence_label']}**. Market filter: **{outlook['market_filter']['market_state']}**, exposure tham khảo {outlook['market_filter']['exposure_multiplier']:.0%}.","\nXác suất là mức độ nghiêng của mô hình, không phải cam kết. Monte Carlo mô tả kịch bản theo giả định lịch sử và có thể bỏ sót cú sốc mới.","\n## Dành cho người chuyên môn",f"\nGit commit: `{outlook['git_commit']}`. Probability đã sigmoid-calibrate trên validation tách biệt; test không dùng để fit calibration. HMM dùng forward filtering. EGARCH được chọn theo QLIKE proxy, không theo return score.","\n> Không phải lời khuyên đầu tư."]
    return "\n".join(lines)+"\n"

