"""LINE 指令接收伺服器（Messaging API webhook）。

只綁 127.0.0.1，由 ngrok 之類的通道把 LINE 的 webhook 轉進來。安全措施：
1. 每個請求都用 Channel secret 驗證 X-Line-Signature，簽章不對一律 403
2. 只處理你本人（LINE_USER_ID）的文字訊息，其他人傳的一律不回覆
3. 只能設定規則、查詢、記帳，沒有任何下單功能

憑證都從環境變數讀：LINE_CHANNEL_SECRET、LINE_CHANNEL_ACCESS_TOKEN、LINE_USER_ID。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

from .config import ROOT, load_config
from . import users
from .line_commands import handle
from .notify import _env, split_message

REPLY_URL = "https://api.line.me/v2/bot/message/reply"
_lock = threading.Lock()   # 同一時間只處理一則會改檔案的指令


def valid_signature(body: bytes, signature: str, secret: str) -> bool:
    if not secret or not signature:
        return False
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).digest()
    return hmac.compare_digest(base64.b64encode(digest).decode(), signature)


def text_events(payload: dict) -> list[tuple[str, str, str]]:
    """回傳 [(userId, replyToken, 文字)]，只保留一對一聊天的文字訊息。"""
    out = []
    for ev in payload.get("events", []):
        src = ev.get("source", {})
        if (ev.get("type") == "message" and ev.get("message", {}).get("type") == "text"
                and src.get("type") == "user" and src.get("userId") and ev.get("replyToken")):
            out.append((src["userId"], ev["replyToken"], ev["message"]["text"]))
    return out


def dispatch(cfg: dict, user_id: str, text: str, handle_fn=handle) -> str | None:
    """本人的訊息：執行指令並回傳要回覆的文字。其他人：不回覆（回傳 None）、只記一筆忽略。"""
    profile = users.get_profile(cfg, user_id)
    if profile is None:
        _log(cfg, "忽略非本人的訊息")
        return None
    res = handle_fn(text, load_config(), profile)
    _log(cfg, f"指令：{text[:60]}" + ("（已修改）" if res.changed else ""))
    return res.reply


def reply(token: str, text: str, post=requests.post) -> None:
    body = {"replyToken": token, "messages": [{"type": "text", "text": t} for t in split_message(text)]}
    post(REPLY_URL, json=body, headers={"Authorization": f"Bearer {_env('LINE_CHANNEL_ACCESS_TOKEN')}"}, timeout=15)


def _log(cfg, msg: str) -> None:
    p = ROOT / cfg["data_dir"] / "line_bot.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")


def make_handler(cfg: dict, handle_fn=handle, reply_fn=reply):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # 不把請求內容印到主控台
            pass

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write("台股研究工具 LINE 指令伺服器運作中".encode("utf-8"))

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if not valid_signature(body, self.headers.get("X-Line-Signature", ""), _env("LINE_CHANNEL_SECRET")):
                _log(cfg, "拒絕：簽章不符")
                self.send_response(403)
                self.end_headers()
                return
            self.send_response(200)          # 先回 200，LINE 才不會重送
            self.end_headers()
            try:
                payload = json.loads(body or b"{}")
            except ValueError:
                return
            for uid, token, text in text_events(payload):
                with _lock:
                    try:
                        msg = dispatch(cfg, uid, text, handle_fn)
                        if msg is None:
                            continue
                    except Exception as e:  # 指令失敗也要回覆，不讓伺服器掛掉
                        msg = f"處理失敗：{type(e).__name__}：{e}"
                        _log(cfg, f"失敗：{text[:60]} → {type(e).__name__}")
                try:
                    reply_fn(token, msg)
                except requests.RequestException as e:
                    _log(cfg, f"回覆失敗：{type(e).__name__}")
    return Handler


def serve(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    lb = cfg.get("line_bot", {})
    host, port = "127.0.0.1", int(lb.get("port", 8787))
    missing = [n for n in ("LINE_CHANNEL_SECRET", "LINE_CHANNEL_ACCESS_TOKEN", "LINE_USER_ID") if not _env(n)]
    if missing:
        raise SystemExit("缺少環境變數：" + "、".join(missing) + "（設定方式見 docs/LINE指令設定.md）")
    httpd = ThreadingHTTPServer((host, port), make_handler(cfg))
    _log(cfg, f"啟動 http://{host}:{port}")
    print(f"LINE 指令伺服器：http://{host}:{port}（Ctrl+C 結束）")
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
