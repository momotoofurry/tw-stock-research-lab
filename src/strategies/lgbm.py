"""LightGBM 預測未來漲跌，walk-forward 滾動訓練。

每一折：訓練列 [s - gap - train_size, s - gap)，測試列 [s, s + test_size)。
標籤用到第 t + horizon 天的收盤，所以 gap 必須 >= horizon：訓練期最後一天的標籤才會早於測試期第一天。

降低換手（成本是主要虧損來源）：
- horizon / hurdle：預測「未來 horizon 天報酬 > hurdle」而不是隔日漲跌
- threshold / exit_threshold：機率 > threshold 進場、< exit_threshold 出場，中間維持原部位（遲滯）
- min_hold：進場後至少持有幾天
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from ..features import build_features, make_label


@dataclass
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_splits(n: int, train_size: int, test_size: int, gap: int, horizon: int = 1) -> list[Fold]:
    if gap < max(horizon, 1):
        raise ValueError(f"gap ({gap}) 必須 >= horizon ({horizon}) 且 >= 1：標籤用到未來 {horizon} 天，"
                         "gap 不夠會讓訓練標籤偷看到測試期")
    folds = []
    s = train_size + gap
    while s < n:
        train = np.arange(s - gap - train_size, s - gap)
        test = np.arange(s, min(s + test_size, n))
        folds.append(Fold(train, test))
        s += test_size
    return folds


def hysteresis(proba: pd.Series, enter: float, exit_: float, min_hold: int = 1) -> pd.Series:
    """機率 > enter 進場；持有滿 min_hold 天後機率 < exit_ 才出場。只用當天以前的機率。"""
    p = proba.to_numpy()
    out = np.full(len(p), np.nan)
    pos, held = 0, 0
    for t in range(len(p)):
        if np.isnan(p[t]):
            continue
        if pos == 0 and p[t] > enter:
            pos, held = 1, 0
        elif pos == 1:
            held += 1
            if held >= min_hold and p[t] < exit_:
                pos = 0
        out[t] = pos
    return pd.Series(out, index=proba.index, name="signal")


def generate_signals(prices: pd.DataFrame, train_size: int = 500, test_size: int = 60, gap: int = 5,
                     threshold: float = 0.5, params: dict | None = None,
                     feature_fn=build_features, return_details: bool = False,
                     horizon: int = 1, hurdle: float = 0.0, exit_threshold: float | None = None,
                     min_hold: int = 1):
    feats = feature_fn(prices)
    label = make_label(prices, horizon, hurdle).reindex(feats.index)
    valid = feats.notna().all(axis=1)          # 去掉暖機期（長均線還沒算出來）
    X = feats[valid]
    y = label[valid]

    proba = pd.Series(np.nan, index=feats.index, name="proba")
    fold_info = []
    for k, fold in enumerate(walk_forward_splits(len(X), train_size, test_size, gap, horizon)):
        Xtr, ytr = X.iloc[fold.train_idx], y.iloc[fold.train_idx]
        keep = ytr.notna()                     # 只用標籤已知的列
        Xtr, ytr = Xtr[keep], ytr[keep]
        if ytr.nunique() < 2:                  # 訓練期全是同一類，無法訓練
            continue
        model = LGBMClassifier(**(params or {}))
        model.fit(Xtr, ytr.astype(int))
        Xte = X.iloc[fold.test_idx]
        p = model.predict_proba(Xte)[:, 1]
        proba.loc[Xte.index] = p
        yte = y.iloc[fold.test_idx]
        m = yte.notna()
        acc = float(((p[m.values] > threshold) == (yte[m] > 0.5)).mean()) if m.any() else float("nan")
        fold_info.append(dict(fold=k, train_start=Xtr.index[0], train_end=Xtr.index[-1],
                              test_start=Xte.index[0], test_end=Xte.index[-1],
                              n_train=len(Xtr), n_test=len(Xte), accuracy=acc,
                              base_rate=float(yte[m].mean()) if m.any() else float("nan")))

    exit_threshold = threshold if exit_threshold is None else exit_threshold
    sig = hysteresis(proba, threshold, exit_threshold, min_hold)
    if return_details:
        return sig, pd.DataFrame(fold_info), proba
    return sig
