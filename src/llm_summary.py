"""本機 LLM 白話摘要：把每日摘要改寫成 3~5 句白話重點。

原則：只用本機 Ollama（資料不得送出本機）；輸出必須通過數字查核與用語檢查，否則不採用；
任何錯誤都回 (None, 原因)，不拋例外。log 只記時間 / 模型 / 結果 / 耗時 / 字數，不記任何內容。
不預測、不給買賣建議。
"""
from __future__ import annotations

import csv
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from .config import ROOT

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
BANNED_TERMS = ["買進", "賣出", "買入", "賣掉", "加碼", "減碼", "建議", "推薦", "目標價", "應該", "進場", "出場",
                "停損", "停利", "看好", "看壞"]
HEADER = "白話摘要（本機模型產生，數字已逐一比對原文）"
LOG_FIELDS = ["time", "model", "ok", "reason", "latency_s", "n_chars"]

SYSTEM_PROMPT = (
    "你是資料改寫助手。把使用者提供的每日股市摘要改寫成 3 到 5 句繁體中文白話重點。\n"
    "嚴格規則：\n"
    "1. 只能使用原文出現過的數字與股票代號，不得新增、換算或四捨五入任何數字。\n"
    "2. 不推論、不預測、不評論好壞，不給任何買賣或操作建議。\n"
    "3. 只描述原文已經寫出的事實，不加入原文沒有的資訊。\n"
    "4. 直接輸出摘要文字，不要標題、不要條列符號、不要解釋。"
)


class LLMError(Exception):
    pass


def _norm_digits(text: str) -> str:
    """全形轉半形（NFKC），並把常見的負號 / 逗號變體統一。"""
    t = unicodedata.normalize("NFKC", text)
    return t.replace("−", "-").replace("－", "-").replace("，", ",")


_NUM_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?")


def extract_numbers(text: str) -> set[str]:
    """抽出文字中所有數字（整數、小數、千分位、百分比、日期中的數字；負號與前導 + 不算進數值）。

    正規化：去千分位逗號、去 %、去前導 +/-；小數尾端的 0 保留原樣但另外也記錄去尾 0 的形式。
    """
    out = set()
    for m in _NUM_RE.finditer(_norm_digits(text)):
        s = m.group(0).replace(",", "")
        out.add(s)
        if "." in s:
            out.add(s.rstrip("0").rstrip("."))
    return out


def check_numbers(source: str, summary: str) -> tuple[bool, list[str]]:
    """摘要中每個數字都必須出現在原文的數字集合；回傳 (通過, 不符的數字清單)。"""
    src = extract_numbers(source)
    bad = []
    for m in _NUM_RE.finditer(_norm_digits(summary)):
        s = m.group(0).replace(",", "")
        forms = {s}
        if "." in s:
            forms.add(s.rstrip("0").rstrip("."))
        if not (forms & src) and s not in bad:
            bad.append(s)
    return (not bad), bad


def check_terms(summary: str) -> list[str]:
    return [w for w in BANNED_TERMS if w in summary]


def strip_think(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^.*?</think>", "", text, flags=re.S)   # 只剩結尾標籤的情況
    return text.strip()


def _check_host(host: str) -> str:
    u = urlparse(host if "://" in host else "http://" + host)
    name = (u.hostname or "").lower()
    if name not in LOCAL_HOSTS:
        raise LLMError(f"host 不是本機（{name or host}）：資料不得送出本機")
    return host.rstrip("/")


def _post_chat(cfg_llm: dict, source_text: str) -> str:
    import requests
    base = _check_host(str(cfg_llm.get("host", "http://127.0.0.1:11434")))
    body = {
        "model": cfg_llm["model"],
        "stream": False,
        "keep_alive": cfg_llm.get("keep_alive", "5m"),
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "請改寫以下每日摘要：\n\n" + source_text},
        ],
    }
    r = requests.post(base + "/api/chat", json=body, timeout=float(cfg_llm.get("timeout_seconds", 180)))
    r.raise_for_status()
    return str(r.json()["message"]["content"])


def _log(cfg: dict, model: str, ok: bool, reason: str, latency: float, n_chars: int) -> None:
    """只追加：不記摘要或原文內容。寫不進去也不影響主流程。"""
    try:
        p = Path(cfg.get("llm", {}).get("log_path") or ROOT / "data" / "llm_log.csv")
        p.parent.mkdir(parents=True, exist_ok=True)
        new = not p.exists()
        with open(p, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(LOG_FIELDS)
            w.writerow([datetime.now().isoformat(timespec="seconds"), model, int(ok), reason,
                        f"{latency:.1f}", n_chars])
    except Exception:
        pass


def summarize(cfg: dict, source_text: str) -> tuple[str | None, str]:
    """回傳 (摘要或 None, 原因)。通過數字查核、用語檢查、長度上限才回摘要；絕不拋例外。"""
    lc = cfg.get("llm", {}) or {}
    if not lc.get("enabled", False):
        return None, "disabled"
    model = str(lc.get("model", ""))
    t0 = time.time()
    summary, reason, ok = None, "", False
    try:
        raw = _post_chat(lc, source_text)
        text = strip_think(raw)
        max_chars = int(lc.get("max_chars", 300))
        if not text:
            reason = "empty"
        elif len(text) > max_chars:
            reason = f"too_long（{len(text)} > {max_chars}）"
        elif check_terms(text):
            reason = "banned_terms：" + "、".join(check_terms(text))
        else:
            good, bad = check_numbers(source_text, text)
            if not good:
                reason = "numbers_mismatch：" + "、".join(bad)
            else:
                summary, reason, ok = text, "ok", True
    except LLMError as e:
        reason = f"rejected：{e}"
    except Exception as e:
        reason = f"error：{type(e).__name__}"
    _log(cfg, model, ok, reason.split("：")[0].split("（")[0] if not ok else reason, time.time() - t0,
         len(summary) if summary else 0)
    return summary, reason
