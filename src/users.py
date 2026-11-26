"""使用者（目前只有管理者本人 = 環境變數 LINE_USER_ID）。

集中定義持股檔、交易紀錄、復原備份、警示狀態的路徑。其他 LINE 使用者傳的訊息一律忽略。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ROOT
from .notify import _env


@dataclass
class Profile:
    user_id: str
    name: str
    data_dir: Path

    @property
    def portfolio_path(self) -> Path:
        return ROOT / "portfolio.yaml"

    @property
    def trades_path(self) -> Path:
        return self.data_dir / "trades.csv"

    @property
    def undo_path(self) -> Path:
        return self.data_dir / "portfolio_undo.yaml"

    @property
    def alert_state_path(self) -> Path:
        return self.data_dir / "alert_state.json"


def owner_profile(cfg: dict) -> Profile:
    d = Path(cfg["data_dir"])
    return Profile(_env("LINE_USER_ID").strip(), "我", d if d.is_absolute() else ROOT / d)


def get_profile(cfg: dict, user_id: str) -> Profile | None:
    """只有本人才有 Profile；其他人回傳 None（訊息會被忽略）。"""
    owner = _env("LINE_USER_ID").strip()
    return owner_profile(cfg) if user_id and user_id == owner else None
