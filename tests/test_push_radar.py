"""LINE 個股走向訊息分段、選股雷達訊息格式。不連網。"""
import pandas as pd

from src import push, radar


def test_analysis_messages_chunk_by_four(monkeypatch):
    monkeypatch.setattr(push, "stock_message", lambda sid, cfg: f"【{sid}】內容")
    msgs = push.analysis_messages([str(i) for i in range(1, 7)], {}, title="持股走向")
    assert len(msgs) == 2
    assert msgs[0].startswith("【持股走向】（1）") and "【4】" in msgs[0] and "【5】" not in msgs[0]
    assert all("不是買賣建議" in m for m in msgs)
    assert push.analysis_messages(["1"], {})[0].startswith("【個股走向】\n")


def test_radar_message_marks_rule_output(monkeypatch):
    monkeypatch.setattr(radar, "rule_record", lambda: ["開發期", "  策略：年化 30.6%"])
    table = pd.DataFrame([dict(stock_id="1111", 名稱="甲", 產業="半導體業", mom=1.2, yoy3=0.3, eps_ttm=5.0),
                          dict(stock_id="2222", 名稱="乙", 產業="電子工業", mom=0.8, yoy3=0.1, eps_ttm=2.0)])
    r = dict(date=pd.Timestamp("2026-10-29"), last_reb=pd.Timestamp("2026-09-01"), held=["1111", "3333"],
             now=["1111", "2222"], table=table, names={"1111": "甲", "2222": "乙", "3333": "丙"}, n=1400,
             next_reb=pd.Timestamp("2026-12-01"))
    text = radar.radar_message(r)
    assert "1. 1111 甲" in text and "2. 2222 乙" in text
    assert "新進 2222" in text and "移出 3333" in text
    assert "不是推薦" in text and "2026-12-01" in text
