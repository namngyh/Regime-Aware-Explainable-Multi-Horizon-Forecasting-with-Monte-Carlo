import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def _safe(value):
    if value is None:
        return np.nan
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return np.nan
    return value


def regression_metrics(y_true, y_pred):
    if y_pred is None:
        return {}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    corr = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred),
        "spearman_ic": _safe(corr),
        "pred_return_mean": np.mean(y_pred),
        "pred_return_std": np.std(y_pred),
    }


def classification_metrics(y_true, y_pred, y_score=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = np.nan
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = np.nan
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def financial_metrics(dates, strategy_returns, benchmark_returns, signals):
    sr = pd.Series(np.asarray(strategy_returns), index=pd.to_datetime(dates)).fillna(0.0)
    br = pd.Series(np.asarray(benchmark_returns), index=pd.to_datetime(dates)).fillna(0.0)
    sig = pd.Series(np.asarray(signals), index=sr.index).fillna(0.0)
    equity = (1 + sr).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    days = max((sr.index.max() - sr.index.min()).days, 1)
    years = days / 365.25
    total_return = equity.iloc[-1] - 1 if len(equity) else np.nan
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 and len(equity) else np.nan
    ann_vol = sr.std() * np.sqrt(252)
    downside = sr[sr < 0].std() * np.sqrt(252)
    sharpe = sr.mean() / sr.std() * np.sqrt(252) if sr.std() > 0 else np.nan
    sortino = sr.mean() / downside * np.sqrt(252) if downside and downside > 0 else np.nan
    beta = sr.cov(br) / br.var() if br.var() > 0 else np.nan
    alpha_daily = sr.mean() - beta * br.mean() if np.isfinite(beta) else np.nan
    active = sr - br
    info_ratio = active.mean() / active.std() * np.sqrt(252) if active.std() > 0 else np.nan
    gross_profit = sr[sr > 0].sum()
    gross_loss = -sr[sr < 0].sum()
    return {
        "strategy_total_return": total_return,
        "strategy_cagr": cagr,
        "strategy_ann_vol": ann_vol,
        "strategy_sharpe": sharpe,
        "strategy_sortino": sortino,
        "strategy_max_drawdown": drawdown.min(),
        "strategy_calmar": cagr / abs(drawdown.min()) if drawdown.min() < 0 else np.nan,
        "strategy_win_rate_daily": (sr > 0).mean(),
        "strategy_profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.nan,
        "strategy_exposure": sig.mean(),
        "strategy_turnover": sig.diff().abs().fillna(0).mean(),
        "strategy_beta_to_buy_hold": beta,
        "strategy_alpha_annualized": alpha_daily * 252 if np.isfinite(alpha_daily) else np.nan,
        "strategy_information_ratio": info_ratio,
        "buy_hold_total_return": (1 + br).prod() - 1,
        "buy_hold_sharpe": br.mean() / br.std() * np.sqrt(252) if br.std() > 0 else np.nan,
    }
