"""LINE 推播（LINE Messaging API）。LINE Notify 已於 2025-03-31 停止服務，所以改用官方帳號 + Messaging API。

憑證只從環境變數讀取，不寫進任何檔案：
- LINE_CHANNEL_ACCESS_TOKEN：LINE Developers → 你的 Messaging API channel → Channel access token（long-lived）
- LINE_USER_ID：你自己的 user ID（LINE Developers → Basic settings → Your user ID，U 開頭）
沒有設定時只印出訊息（dry run），不會送出。
"""
from __future__ import annotations

import os
import re

import requests

PUSH_URL = "https://api.line.me/v2/bot/message/push"
MAX_CHARS = 5000  # LINE 單則文字訊息上限


class NotifyError(RuntimeError):
    pass


def _env(name: str) -> str:
    """先看目前行程的環境變數；Windows 上沒有時再讀使用者環境變數（setx 設定的值，排程剛啟動時可能還沒帶入）。"""
    v = os.environ.get(name, "")
    if not v and os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                v = str(winreg.QueryValueEx(k, name)[0])
        except OSError:
            v = ""
    return v


def line_configured() -> bool:
    return bool(_env("LINE_CHANNEL_ACCESS_TOKEN") and _env("LINE_USER_ID"))


def split_message(text: str, limit: int = MAX_CHARS) -> list[str]:
    """依行切成不超過 limit 字的段落；LINE 一次 push 最多 5 則。"""
    parts, cur = [], ""
    for line in text.splitlines():
        while len(line) > limit:
            parts.append(line[:limit])
            line = line[limit:]
        if len(cur) + len(line) + 1 > limit:
            parts.append(cur.rstrip("\n"))
            cur = ""
        cur += line + "\n"
    if cur.strip():
        parts.append(cur.rstrip("\n"))
    return parts[:5]


def send_line(text: str, timeout: int = 15, post=requests.post) -> bool:
    """推播給管理者（LINE_USER_ID）。送出成功回傳 True；未設定憑證時印出內容並回傳 False。"""
    if not line_configured():
        print("[notify] 未設定 LINE_CHANNEL_ACCESS_TOKEN / LINE_USER_ID，只印出不送出：\n" + text)
        return False
    return send_line_to(_env("LINE_USER_ID").strip(), text, timeout, post)


def send_line_to(to: str, text: str, timeout: int = 15, post=requests.post) -> bool:
    """推播給指定的 LINE user ID。失敗拋出 NotifyError（訊息不含 token）。"""
    token = _env("LINE_CHANNEL_ACCESS_TOKEN")
    if not token:
        print("[notify] 未設定 LINE_CHANNEL_ACCESS_TOKEN，只印出不送出：\n" + text)
        return False
    if not re.fullmatch(r"U[0-9a-f]{32}", to):
        raise NotifyError(f"LINE user ID 格式不對（目前 {len(to)} 個字元）：應為 U 開頭 + 32 個英數字、共 33 個字元，"
                          "在 LINE Developers → 你的 channel → Basic settings 最下方「Your user ID」")
    body = {"to": to, "messages": [{"type": "text", "text": t} for t in split_message(text)]}
    try:
        r = post(PUSH_URL, json=body, headers={"Authorization": f"Bearer {token}"}, timeout=timeout)
    except requests.RequestException as e:
        raise NotifyError(f"LINE 推播連線失敗：{type(e).__name__}") from None
    if r.status_code != 200:
        msg = ""
        try:
            msg = r.json().get("message", "")
        except ValueError:
            pass
        hint = {401: "token 無效或過期", 400: "user ID 錯誤或沒有加官方帳號好友",
                429: "超過每月訊息額度或頻率限制"}.get(r.status_code, "")
        raise NotifyError(f"LINE 推播失敗 HTTP {r.status_code} {msg} {hint}".strip())
    return True
