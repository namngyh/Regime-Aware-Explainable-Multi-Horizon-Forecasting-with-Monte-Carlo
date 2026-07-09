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
