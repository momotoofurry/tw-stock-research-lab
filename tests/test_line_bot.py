"""LINE 指令：簽章、身分過濾、申請核准、指令解析、每人資料分開、復原。不連網。"""
import base64
import hashlib
import hmac
import json

import pytest

from src import line_bot, line_commands, portfolio as pfm, users

OWNER = "U" + "a" * 32
STRANGER = "U" + "c" * 32
NAMES = {"2317": "鴻海", "2303": "聯電", "1216": "統一"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(users, "ROOT", tmp_path)
    monkeypatch.setattr(users, "_env", lambda n: OWNER if n == "LINE_USER_ID" else "x")
    cfg = {"data_dir": str(tmp_path / "data"), "costs": dict(commission_rate=0.001425, commission_discount=1.0,
                                                           min_commission=20, sell_tax_rate=0.003, price_limit=0.1)}
    owner = users.owner_profile(cfg)
    pf = pfm.load_portfolio(owner.portfolio_path)
    pf["holdings"] = [dict(stock_id="2317", shares=1000, avg_cost=100.0), dict(stock_id="2303", shares=2000, avg_cost=50.0)]
    pfm.save_portfolio(pf, owner.portfolio_path)
    return cfg, owner


def run(text, cfg, profile):
    lookup = lambda sid: NAMES.get(sid, "")  # noqa: E731
    return line_commands.handle(text, cfg, profile, lookup=lookup, holdings_fn=lambda pf, c: "HOLDINGS")


def test_signature():
    body = b'{"events":[]}'
    sig = base64.b64encode(hmac.new(b"secret", body, hashlib.sha256).digest()).decode()
    assert line_bot.valid_signature(body, sig, "secret")
    assert not line_bot.valid_signature(body, sig, "other")
    assert not line_bot.valid_signature(body + b" ", sig, "secret")
    assert not line_bot.valid_signature(body, "", "secret")


def test_text_events_filters_groups_and_non_text():
    payload = {"events": [
        {"type": "message", "replyToken": "t1", "source": {"type": "user", "userId": OWNER},
         "message": {"type": "text", "text": "規則"}},
        {"type": "message", "replyToken": "t2", "source": {"type": "group", "groupId": "G", "userId": OWNER},
         "message": {"type": "text", "text": "規則"}},
        {"type": "message", "replyToken": "t3", "source": {"type": "user", "userId": OWNER},
         "message": {"type": "sticker"}},
        {"type": "follow", "replyToken": "t4", "source": {"type": "user", "userId": STRANGER}},
    ]}
    assert line_bot.text_events(payload) == [(OWNER, "t1", "規則")]


def test_global_rules_set_cancel_and_undo(env):
    cfg, owner = env
    r = run("停損 10%", cfg, owner)
    assert r.changed and "10%" in r.reply
    assert pfm.load_portfolio(owner.portfolio_path)["defaults"]["stop_loss_pct"] == pytest.approx(-0.10)
    run("移動停損 15", cfg, owner)
    run("均線 60 日線", cfg, owner)
    run("權重上限 40%", cfg, owner)
    d = pfm.load_portfolio(owner.portfolio_path)["defaults"]
    assert d["trailing_stop_pct"] == pytest.approx(-0.15) and d["ma_break"] == 60 and d["max_weight"] == pytest.approx(0.4)
    run("取消 均線", cfg, owner)
    assert pfm.load_portfolio(owner.portfolio_path)["defaults"]["ma_break"] is None
    assert run("復原", cfg, owner).changed
    assert pfm.load_portfolio(owner.portfolio_path)["defaults"]["ma_break"] == 60
    assert "沒有可以復原" in run("復原", cfg, owner).reply


def test_per_stock_rules_by_name_and_code(env):
    cfg, owner = env
    run("聯電 停損價 65", cfg, owner)
    run("2317 移動停損 20%", cfg, owner)
    hs = {h["stock_id"]: h for h in pfm.load_portfolio(owner.portfolio_path)["holdings"]}
    assert hs["2303"]["stop_loss"] == 65 and hs["2317"]["trailing_stop_pct"] == pytest.approx(-0.2)
    run("鴻海 取消 移動停損", cfg, owner)
    assert pfm.load_portfolio(owner.portfolio_path)["holdings"][0]["trailing_stop_pct"] is None
    assert "持股裡找不到" in run("統一 停損價 1400", cfg, owner).reply       # 這個測試持股沒有統一
    assert "只能設成全體規則" in run("聯電 組合回撤 10%", cfg, owner).reply
    assert "要加上股票" in run("停損價 65", cfg, owner).reply


def test_validation_and_help(env):
    cfg, owner = env
    assert "介於" in run("停損 150%", cfg, owner).reply
    assert "請加上數值" in run("停損", cfg, owner).reply
    assert "看不懂" in run("幫我選股", cfg, owner).reply
    assert "停損 10%" in run("說明", cfg, owner).reply
    assert run("持股", cfg, owner).reply == "HOLDINGS"
    assert "停損（低於成本）" in run("規則", cfg, owner).reply


def test_trade_command(env):
    cfg, owner = env
    r = run("買 2317 50 450", cfg, owner)
    assert r.changed and "1,050 股" in r.reply
    h = next(x for x in pfm.load_portfolio(owner.portfolio_path)["holdings"] if x["stock_id"] == "2317")
    assert h["shares"] == 1050
    assert "不能賣" in run("賣 2317 5000 450", cfg, owner).reply
    assert owner.trades_path.exists()


def test_only_owner_is_served_strangers_get_no_reply(env, monkeypatch):
    cfg, owner = env
    handle = lambda text, c, p: run(text, cfg, p)  # noqa: E731
    monkeypatch.setattr(line_bot, "load_config", lambda: cfg)
    assert line_bot.dispatch(cfg, STRANGER, "停損 10%", handle) is None          # 不回覆、不執行
    assert pfm.load_portfolio(owner.portfolio_path)["defaults"]["stop_loss_pct"] is None
    assert users.get_profile(cfg, STRANGER) is None
    assert "已設定" in line_bot.dispatch(cfg, OWNER, "停損 10%", handle)
    assert pfm.load_portfolio(owner.portfolio_path)["defaults"]["stop_loss_pct"] == pytest.approx(-0.10)


def test_webhook_rejects_bad_signature_and_replies(env, monkeypatch):
    import io
    cfg, owner = env
    monkeypatch.setattr(line_bot, "_env", lambda n: {"LINE_CHANNEL_SECRET": "s", "LINE_USER_ID": OWNER}.get(n, "x"))
    monkeypatch.setattr(line_bot, "load_config", lambda: cfg)
    replies = []
    H = line_bot.make_handler(cfg, handle_fn=lambda t, c, p: line_commands.Result("ok:" + t),
                              reply_fn=lambda tok, msg: replies.append((tok, msg)))
    body = json.dumps({"events": [{"type": "message", "replyToken": "tk", "source": {"type": "user", "userId": OWNER},
                                   "message": {"type": "text", "text": "規則"}}]}).encode()

    def call(sig):
        h = H.__new__(H)
        h.rfile, h.wfile = io.BytesIO(body), io.BytesIO()
        h.headers = {"Content-Length": str(len(body)), "X-Line-Signature": sig}
        codes = []
        h.send_response = codes.append
        h.end_headers = lambda: None
        h.do_POST()
        return codes[0]

    assert call("bad") == 403 and replies == []
    good = base64.b64encode(hmac.new(b"s", body, hashlib.sha256).digest()).decode()
    assert call(good) == 200 and replies == [("tk", "ok:規則")]


def test_trend_and_checkup_commands(env):
    cfg, owner = env
    calls = []
    fn = lambda kind, sid, c: calls.append((kind, sid)) or f"{kind}:{sid}"  # noqa: E731
    lookup = lambda sid: NAMES.get(sid, "")  # noqa: E731
    r1 = line_commands.handle("趨勢 鴻海", cfg, owner, lookup=lookup, analysis_fn=fn)
    r2 = line_commands.handle("體檢 2303", cfg, owner, lookup=lookup, analysis_fn=fn)
    assert (r1.reply, r2.reply) == ("趨勢:2317", "體檢:2303") and not r1.changed
    assert "趨勢 2317" in line_commands.HELP
