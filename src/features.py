import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def make_features(raw: pd.DataFrame, horizons=(5, 20, 60)) -> tuple[pd.DataFrame, list[str]]:
    df = raw.copy()
    df["ret_1"] = df["close"].pct_change()
    df["log_ret_1"] = np.log(df["close"]).diff()
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["gap_pct"] = df["open"] / df["close"].shift(1) - 1
    df["volume_chg"] = df["volume"].pct_change()

    for window in (3, 5, 10, 20, 60, 120):
        df[f"ret_{window}"] = df["close"].pct_change(window)
        df[f"mom_{window}"] = df["close"] / df["close"].shift(window) - 1
        df[f"vol_{window}"] = df["log_ret_1"].rolling(window).std() * np.sqrt(252)
        df[f"sma_ratio_{window}"] = df["close"] / df["close"].rolling(window).mean() - 1
        df[f"volume_z_{window}"] = (
            df["volume"] - df["volume"].rolling(window).mean()
        ) / df["volume"].rolling(window).std()

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_bullish"] = (df["macd"] > df["macd_signal"]).astype(int)
    df["rsi_14"] = rsi(df["close"], 14)
    df["daily_return_next"] = df["close"].pct_change().shift(-1)

    for horizon in horizons:
        df[f"future_return_{horizon}d"] = df["close"].shift(-horizon) / df["close"] - 1
        df[f"future_up_{horizon}d"] = (df[f"future_return_{horizon}d"] > 0).astype(int)

    feature_cols = [
        col
        for col in df.columns
        if col
        not in {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "daily_return_next",
            *[f"future_return_{h}d" for h in horizons],
            *[f"future_up_{h}d" for h in horizons],
        }
    ]
    df = df.replace([np.inf, -np.inf], np.nan)
    return df, feature_cols


def chronological_split(df: pd.DataFrame, train_ratio=0.70, valid_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:valid_end].copy(),
        df.iloc[valid_end:].copy(),
    )


def make_raemf_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build causal, non-redundant OHLCV features for the RAEMF-MC pipeline."""
    df = raw.copy().sort_values("date").reset_index(drop=True)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df else None
    log_ret = np.log(close).diff()
    df["ret_1"] = close.pct_change()
    df["log_ret_1"] = log_ret
    for window in (5, 10, 20, 40, 60, 120):
        df[f"ret_{window}"] = close.pct_change(window)
        df[f"vol_{window}"] = log_ret.rolling(window).std() * np.sqrt(252)
        df[f"price_to_sma_{window}"] = close / close.rolling(window).mean() - 1
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["ema_spread"] = ema12 / ema26 - 1
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_hist_slope"] = df["macd_hist"].diff(5) / 5
    df["rsi"] = rsi(close, 14)
    true_range = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    plus_dm = high.diff().where((high.diff() > -low.diff()) & (high.diff() > 0), 0.0)
    minus_dm = (-low.diff()).where((-low.diff() > high.diff()) & (-low.diff() > 0), 0.0)
    atr = true_range.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(14).mean() / atr.replace(0, np.nan)
    df["adx"] = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).rolling(14).mean()
    df["atr_normalized"] = atr / close
    df["trend_strength"] = df["ret_60"] / (df["vol_60"] / np.sqrt(252) * np.sqrt(60) + 1e-8)
    df["up_day_ratio_20"] = (df["ret_1"] > 0).rolling(20).mean()
    df["up_day_ratio_60"] = (df["ret_1"] > 0).rolling(60).mean()
    df["distance_to_52w_high"] = close / close.rolling(252).max() - 1
    df["distance_to_52w_low"] = close / close.rolling(252).min() - 1
    df["return_sign_persistence"] = np.sign(df["ret_1"]).rolling(20).mean()
    df["trend_consistency"] = df["ret_1"].rolling(20).mean() / (df["ret_1"].rolling(20).std() + 1e-8)
    for window in (20, 60):
        df[f"downside_vol_{window}"] = log_ret.where(log_ret < 0, 0).rolling(window).std() * np.sqrt(252)
    df["upside_semivariance"] = log_ret.clip(lower=0).pow(2).rolling(60).mean()
    df["downside_semivariance"] = log_ret.clip(upper=0).pow(2).rolling(60).mean()
    df["rolling_skewness"] = log_ret.rolling(60).skew()
    df["rolling_kurtosis"] = log_ret.rolling(60).kurt()
    df["parkinson_volatility"] = np.sqrt(np.log(high / low).pow(2).rolling(20).mean() / (4 * np.log(2))) * np.sqrt(252)
    df["range_volatility"] = ((high - low) / close).rolling(20).mean()
    df["volatility_of_volatility"] = df["vol_20"].rolling(60).std()
    drawdown = close / close.cummax() - 1
    for window in (20, 60, 120):
        df[f"rolling_max_drawdown_{window}"] = (close / close.rolling(window).max() - 1).rolling(window).min()
    underwater = drawdown < 0
    groups = (~underwater).cumsum()
    df["drawdown_duration"] = underwater.groupby(groups).cumsum()
    df["recovery_ratio"] = df["ret_60"] / (df["rolling_max_drawdown_60"].abs() + 1e-8)
    if volume is not None and volume.notna().any():
        df["volume_zscore"] = (volume - volume.rolling(60).mean()) / volume.rolling(60).std()
        df["volume_ratio_5"] = volume / volume.rolling(5).mean()
        df["volume_ratio_20"] = volume / volume.rolling(20).mean()
        df["volume_trend"] = volume.rolling(5).mean() / volume.rolling(20).mean() - 1
        df["price_volume_divergence"] = df["ret_20"] - volume.pct_change(20).replace([np.inf, -np.inf], np.nan)
        df["signed_volume_proxy"] = np.sign(df["ret_1"]) * np.log1p(volume)
        high_volume = volume > volume.rolling(60).quantile(0.75)
        df["high_volume_up_day_ratio"] = (high_volume & (df["ret_1"] > 0)).rolling(60).mean()
        df["high_volume_down_day_ratio"] = (high_volume & (df["ret_1"] < 0)).rolling(60).mean()
        df["volume_volatility"] = np.log1p(volume).diff().rolling(20).std()
        df["volume_confirmation_score"] = np.sign(df["ret_20"]) * df["volume_trend"]
    excluded = {"date", "open", "high", "low", "close", "volume"}
    features = [c for c in df.columns if c not in excluded]
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    return df, features
