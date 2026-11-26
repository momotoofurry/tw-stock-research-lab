"""研究報告的重現資訊：設定雜湊、資料截止日、程式版本（git commit）、套件版本、產生時間。

只記錄設定檔內容的雜湊與版本號，不記錄任何環境變數或憑證。
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path

from .config import ROOT


def config_hash(cfg: dict) -> str:
    """整份設定的 8 碼雜湊（排除只影響執行方式、不影響結果的 finmind 連線設定）。"""
    body = {k: v for k, v in cfg.items() if k != "finmind"}
    return hashlib.sha1(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:8]


def git_commit() -> str:
    """目前的 commit（短碼）；有未提交的修改加上「+未提交修改」；不是 git repo 就回傳「無版本控制」。"""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                             timeout=10)
        if sha.returncode != 0:
            return "無版本控制"
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return sha.stdout.strip() + ("+未提交修改" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "無版本控制"


def package_versions() -> str:
    import numpy
    import pandas
    out = [f"python {platform.python_version()}", f"pandas {pandas.__version__}", f"numpy {numpy.__version__}"]
    try:
        import lightgbm
        out.append(f"lightgbm {lightgbm.__version__}")
    except ImportError:
        pass
    return "、".join(out)


def run_info(cfg: dict, data_end=None, data_hash: str | None = None) -> str:
    end = getattr(data_end, "date", lambda: data_end)() if data_end is not None else "未知"
    dh = f"、資料快照 {data_hash}" if data_hash else ""
    return (f"設定 {config_hash(cfg)}、資料截止 {end}{dh}、程式 {git_commit()}、{package_versions()}、"
            f"產生於 {datetime.now():%Y-%m-%d %H:%M}")


def data_manifest(cfg: dict, stock_ids, out_path) -> str:
    """把研究用到的快取檔（日K、事件、營收、財報）逐檔算 SHA1，寫成 out_path（JSON），回傳整體 8 碼雜湊。
    上游修正歷史資料或重新下載後，同一份程式與設定的結果不同時，可以用這份清單找出是哪些檔案變了。
    .range / .start 只是快取的查詢紀錄（含查詢時間），不列入。"""
    d = ROOT / cfg["data_dir"]
    files = set()
    for sid in stock_ids:
        files |= {f for f in d.glob(f"*_{sid}.*") if f.suffix in (".csv", ".json")}
        files |= {f for f in d.glob(f"*_{sid}_*") if f.suffix in (".csv", ".json")}
    files |= {d / "full_candidates.csv"} & set(d.glob("full_candidates.csv"))
    manifest = {f.name: hashlib.sha1(f.read_bytes()).hexdigest() for f in sorted(files)}
    total = hashlib.sha1(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:8]
    Path(out_path).write_text(json.dumps({"data_hash": total, "files": manifest}, ensure_ascii=False, indent=0),
                              encoding="utf-8")
    return total
