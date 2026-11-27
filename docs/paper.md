# Do Simple Stock-Selection Rules Beat a Passive Benchmark in Taiwan After Realistic Frictions? Evidence from a Pre-Registered Testing Protocol

*Working paper, November 2026. This is a research and simulation project. It contains no order execution and nothing here is investment advice.*

---

## 中文摘要

本研究在台股（上市櫃普通股，2017-07 至 2024-12）用一套事前登記的檢驗流程，評估四個常見的選股規則：動能加基本面品質、營收加速加動能、低本益比加營收成長，以及 AI 依文獻提出的低流動性溢酬。回測模擬了台股真實的交易摩擦：隔日開盤成交、手續費與證交稅、開盤漲跌停無法成交、成交量參與上限、除權息與官方公司行動，股票池也包含已下市股票。

每個規則都要和兩個對照組比較：0050 含息，以及「同股票池、同持股數、同換手率」的配對隨機選股。另外有六項穩健性檢驗，並依至今試過的規則數做 Bonferroni 修正。

結果四個規則全部不通過。年化報酬最高的規則（28.2%）Sharpe 仍低於 0050（0.85 vs 0.91），最大回撤深 23 個百分點，對配對隨機的 p = 0.143。在保留期（2025-01 至 2026-10），它的年化只有 12.3%，同期 0050 是 54.1%。單檔擇時方法（均線交叉、K 線加量能、LightGBM）在 40 檔股票中很少勝過買進持有。多數版本顯著勝過隨機訊號的檔數和純運氣的預期相近；LightGBM 20 日版本略高於運氣水準，但主要是「比隨機進出少輸」，不是賺贏持有。

主要的方法論發現：
- 對照組的設計會決定結論。同一個排名規則，對「單純隨機挑 10 檔」的 p = 0.010，改用配對換手率的隨機組後變成 p = 0.143。
- 只看年化報酬會嚴重誤導。

## Abstract

I evaluate four stock-selection rules on the Taiwan equity market (TWSE and TPEx common stocks, July 2017 to December 2024) under a pre-registered protocol. The backtest models the market's actual frictions:
- next-open execution
- commission with a minimum fee and a sell-side transaction tax
- unfillable limit-up and limit-down opens
- a participation cap on prior-day volume
- officially sourced dividends and corporate actions

The universe includes delisted stocks.

Each rule must beat two controls. The first is the dividend-inclusive 0050 ETF. The second is a matched random-selection portfolio drawn from the same eligible pool, with the same number of holdings and the same turnover. Each rule must also survive six robustness checks, with a Bonferroni correction over every distinct rule ever tried.

**All four rules fail.** The highest-return rule earns 28.2% a year, yet its Sharpe ratio is below the benchmark's (0.85 vs 0.91) and its drawdown is 23 percentage points deeper. Against matched random selection its p-value is 0.143. In the 2025 to 2026 holdout it returned 12.3% a year while the benchmark returned 54.1%. Single-stock timing models (moving-average crossover, candlestick and volume, LightGBM) rarely beat buy-and-hold across 40 stocks. Most produce about as many significant results as chance predicts; the 20-day LightGBM runs exceed that level, but mainly by losing less than random timing.

The central methodological finding is that **the choice of control decides the conclusion**. The same ranking rule has p = 0.010 against naive random 10-stock portfolios and p = 0.143 against turnover-matched random portfolios.

---

## 1. Introduction

Retail-facing strategy marketing usually reports a single backtested annual return. Three problems make that number unreliable:
1. **Implementation frictions.** Fills, fees, taxes, price limits and liquidity are often ignored or simplified.
2. **Look-ahead bias.** Fundamentals are used before they were published, prices are adjusted with future information, or the universe contains only today's survivors.
3. **Multiple testing.** Many variants are tried and only the best is shown (Bailey, Borwein, López de Prado and Zhu, 2014; Harvey, Liu and Zhu, 2016).

This project builds a testing protocol that addresses all three, then applies it honestly, including to rules proposed by the author and by an AI assistant. The question is narrow and practical:

> After realistic Taiwan-market frictions, does a simple, pre-specified stock-selection rule deliver better risk-adjusted returns than holding 0050, by more than luck would explain?

The contribution is the protocol and the negative evidence, not a profitable rule.

## 2. Data

| Item | Detail |
|---|---|
| Source | FinMind API v4 (aggregating TWSE, TPEx and MOPS) |
| Prices | Daily OHLCV, unadjusted, 2016 onward |
| Universe | All TWSE and TPEx common stocks that passed the liquidity screen on at least one rebalance date: 1,637 names, **including delisted stocks** (1,631 loaded) |
| Fundamentals | Monthly revenue and quarterly EPS |
| Corporate actions | Official ex-rights and ex-dividend results, capital reductions, splits and par-value changes, plus dividend announcements to fill gaps in the result table |
| Development period | 2017-07-03 to 2024-12-31 (one year of history needed for revenue growth) |
| Holdout period | 2025-01-01 onward. Hidden by default; every view is logged. Consumed once, on 2026-10-30 |

**Point-in-time alignment.** A fundamental value becomes usable on its statutory deadline plus one day:
- monthly revenue: the 10th of the next month
- quarterly reports: quarter-end + 45 days (financial firms: + 2 months)
- annual reports: March 31

**Known data limitation.** FinMind serves only the latest version of each fundamental value, so restated numbers can replace what was originally announced. Since 2026-11-16 the project appends a vintage snapshot of every newly seen or revised row, so this bias can be measured going forward but not retroactively.

## 3. Backtest Engine

| Rule | Implementation |
|---|---|
| Timing | Signal on the close of day *t*; fill at the **open of *t+1*** |
| Commission | 0.1425%, rounded down to whole TWD, minimum TWD 20 |
| Transaction tax | 0.3% on sells |
| Price limits | Buying at a limit-up open or selling at a limit-down open is skipped and logged. Limits are computed from the reference price and tick-size table |
| Liquidity cap | Shares bought per day ≤ 5% × average volume of the 20 days **before** the fill day |
| Prices used | Signals use forward-adjusted prices. Fills, limits and costs use unadjusted prices |
| Dividends | Credited on the ex-date to holders as of the prior close, identically for strategy and benchmark |
| Corporate actions | Official reference prices only, never inferred from price gaps. For example, an inferred reduction price of 169 vs the official 187 for one shipping stock |
| Delisting | Liquidated at the last close; a zero-recovery stress test is also reported |
| Lot size | Odd lots (1 share) for the multi-stock rules |

Sharpe ratio = mean daily return / standard deviation × √252, with a risk-free rate of 0.

### 3.1 Validating the engine (oracle tests)

The test suite (250 tests) must pass all of the following:
1. A cheating model that sees tomorrow's return must exceed 100% a year. This confirms the pipeline can detect signal.
2. Three deliberately broken engines must each be caught: fill at the same close, omit the transaction tax, shift features one day into the future.
3. Hand-computed costs of a small trade must equal the engine's to the dollar.
4. Random signals must clearly underperform buy-and-hold after costs.
5. In walk-forward training, the last training date must precede the first test date by at least the label horizon.

A **truncation-invariance test** also recomputes features after deleting all future rows and requires the past rows to be identical.

## 4. Testing Protocol

### 4.1 Pre-registration

Each rule is a YAML file written before any result is seen. It contains:
- hypothesis, rationale and a falsification condition
- universe, rebalance interval, number of holdings, lookback
- filters and a ranking factor

The rule is hashed together with every configuration key that affects its result. Any change means a new file, a new hash and a new count. AI-proposed rules must also name a literature source and answer "who pays for this premium". They are subject to a monthly quota and a graveyard rule: a failed AI family cannot be re-proposed with new parameters.

### 4.2 Controls

1. **Benchmark:** 0050 with dividends reinvested.
2. **Matched random selection:** on each rebalance date, draw from that day's eligible pool (stocks passing the same price and liquidity screens; about 490 names on average). Draw the same number of names the rule held. Keep the same number of names the rule kept from its previous holdings, so turnover matches. Run 100 portfolios with fixed seeds (20 for the first run).
3. **Equal-weight eligible pool:** holds every stock that passes the screens.

### 4.3 Pass criteria (all must hold)

| # | Check | Pass condition |
|---|---|---|
| 1 | vs benchmark | Higher Sharpe, and max drawdown no more than 10 pts deeper |
| 2 | vs matched random | Top 5% on both CAGR and Sharpe |
| 3 | Neighbouring parameters (each ±20%) | Sharpe beats benchmark in at least 3/4 of variants |
| 4 | Split sample | Sharpe beats benchmark in both halves |
| 5 | 2× costs | Sharpe still beats benchmark |
| 6 | Delisted holdings → 0 | Sharpe still beats benchmark |
| 7 | Drop top-3 contributors from the universe | Sharpe still beats benchmark |
| 8 | Drop the best calendar year | Sharpe still beats benchmark |
| — | Multiple testing | max(p_CAGR, p_Sharpe) × N < 0.05, where N = distinct rules ever tried |

The trial ledger is append-only. With K = 100 random portfolios, the smallest achievable Bonferroni-adjusted p is N / (K + 1), so the protocol reports when K must grow.

## 5. Results: Stock-Selection Rules

All rules are long-only, equal-weight, filtered by price > TWD 10 and 20-day average traded value > TWD 50 million. The momentum rules follow the intermediate-horizon momentum literature (Jegadeesh and Titman, 1993).

| Rule | Author | Specification | Rebalance / Holdings |
|---|---|---|---|
| `retired_rank` | user (pre-dates the lab) | mom63 > 0, mom126 > 0, revenue YoY (3m) > 0, TTM EPS > 0; rank by mom126 | 63 days / 10 |
| `global_trend_momentum` | user | revenue YoY (3m) > 20%, TTM EPS > 0, mom126 > 0; rank by mom126 | 63 days / 5 |
| `cheap_growth` | user | P/E < 15, revenue YoY (3m) > 10%; rank by lowest P/E | 63 days / 5 |
| `ai_illiquidity_quality` | AI | TTM EPS > 0; rank by **lowest** traded value (Amihud, 2002; Asness et al., 2018) | 126 days / 20 |

### 5.1 Performance (development period, 2017-07 to 2024-12)

| Portfolio | CAGR | Max DD | Sharpe | Turnover / yr |
|---|---|---|---|---|
| **0050 (dividends reinvested)** | **14.9%** | **−30.2%** | **0.91** | 0.1× |
| Equal-weight eligible pool (63-day rebalance)¹ | 13.2% | −35.2% | 0.70 | 1.2× |
| `retired_rank` | 28.2% | −52.9% | 0.85 | 3.3× |
| `global_trend_momentum` | 18.0% | −67.6% | 0.60 | 3.4× |
| `cheap_growth` | −0.7% | −50.1% | 0.09 | 3.2× |
| `ai_illiquidity_quality` | 19.0% | −41.3% | 0.95 | 2.0× |

¹ Matches the first three rules. For `ai_illiquidity_quality` (126-day rebalance) the equal-weight pool earned 13.6%, with max DD −34.9%, Sharpe 0.73 and turnover 0.8×.

### 5.2 Check-by-check outcome

| Check | retired_rank | global_trend_momentum | cheap_growth | ai_illiquidity_quality |
|---|---|---|---|---|
| 1 vs 0050 | ✗ | ✗ | ✗ | ✗ (Sharpe higher, DD 11 pts deeper) |
| 2 matched random (p_CAGR / p_Sharpe) | ✗ 0.048 / 0.143 | ✗ 0.149 / 0.297 | ✗ 0.960 / 0.960 | ✗ 0.089 / 0.059 |
| 3 neighbours | ✗ 3/6 | ✗ 0/8 | ✗ 0/10 | ✗ 3/6 |
| 4 split sample | ✗ | ✗ | ✗ | ✗ |
| 5 2× costs | ✗ | ✗ | ✗ | ✗ |
| 6 delisted → 0 | ✗ | ✗ | ✗ | ✓ |
| 7 drop top 3 | ✗ | ✗ | ✗ | ✗ |
| 8 drop best year | ✗ | ✗ | ✗ | ✗ |
| Bonferroni (N at time of run) | 0.143 (N=1) | 0.594 (N=2) | 1.000 (N=3) | 0.356 (N=4) |
| **Verdict** | **Fail** | **Fail** | **Fail** | **Fail** |

### 5.3 Observations

**High return is not evidence.**
- `retired_rank` nearly doubled the benchmark's CAGR but had a lower Sharpe ratio, and its edge depended on the period. Split-sample Sharpe was 0.60 vs 1.12 in 2017 to 2021 and 1.05 vs 0.70 in 2021 to 2024.
- Removing its best year (2021, +128%) cut its Sharpe to 0.61 vs 0.87 for the benchmark.
- `global_trend_momentum` shows the same pattern: best year 2021 (+133%), Sharpe 0.34 after removal, and 0 of 8 neighbouring parameter sets beat the benchmark.

**The closest candidate is the literature-based one.** `ai_illiquidity_quality` beat 0050 on Sharpe (0.95 vs 0.91) and beat 95% of matched random portfolios on Sharpe. However:
- its drawdown was 11 pts deeper, just outside the 10-pt tolerance
- it lost the advantage under 2× costs (0.89)
- it lost the advantage without its top three contributors (0.73)
- after correcting for four rules tried, p = 0.356

This matches the literature's caveat that the illiquidity premium is partly compensation for trading costs.

**Value plus growth failed outright.** `cheap_growth` underperformed 96% of random portfolios. One likely mechanism, documented before the run: trailing P/E and dividend yield make post-peak cyclicals look cheapest. For example, one shipping stock had a trailing P/E of 1.0 and a yield of 31% in March 2023.

**Small accounts face a different cost structure.** Re-running `ai_illiquidity_quality`'s frozen targets with TWD 20,000 of capital and the TWD 20 minimum fee lowered its CAGR from 19.0% to 13.6%. Costs rose from 1.08% to 5.49% of average equity per year.

## 6. Case Study: The Holdout

`retired_rank` was developed before the lab existed and then viewed once on the 2025-01 to 2026-10 holdout, in a study that used a slightly different universe (1,494 names) and a naive random control.

| Period | Rule CAGR | Rule Sharpe | 0050 CAGR | 0050 Sharpe | Random p |
|---|---|---|---|---|---|
| Development (2017-07 to 2024-12) | 30.6% | 0.89 | 14.9% | 0.91 | 0.010 (naive) |
| **Holdout (2025-01 to 2026-10)** | **12.3%** | **0.48** | **54.1%** | **1.88** | **0.545** |

Two lessons follow.
1. **The control decides the conclusion.** Against naive random 10-stock portfolios, the development result looked highly significant (p = 0.010). Re-tested in the lab against turnover-matched random portfolios, the same rule's p rose to 0.143. A momentum rule with 3.3× turnover is structurally different from a random buy-and-hold basket, and only the matched control isolates selection skill.
2. **Robustness failures predicted out-of-sample failure.** In development the rule already failed the neighbouring-parameter check (2 of 9 beat the benchmark) and depended heavily on one half of the sample. Out of sample, it beat only 46% of random portfolios.

Because the holdout is now spent, the rule is retired, and new rules can earn out-of-sample evidence only through forward paper tracking (Section 8).

## 7. Results: Single-Stock Timing

The same engine was applied to 40 large listed stocks (2018 to 2024, development period only), comparing each timing strategy with buy-and-hold of that stock and with 100 to 200 random signals per stock that match its exposure and turnover. Under the null of no skill, about 2 of 40 stocks (5%) should appear significant by chance.

All eight study runs on record are listed. "v2" rows are re-runs after engine revisions, with 100 rather than 200 random signals per stock.

| Strategy (run) | Beats buy-and-hold (CAGR) | Median excess CAGR | Significant vs random (p < 0.05) | Mean random percentile |
|---|---|---|---|---|
| MA crossover 5/20 (v1) | 2 / 40 | −5.9% | 2 / 40 | 56% |
| MA crossover 5/20 (v2) | 4 / 40 | −5.7% | 3 / 40 | 57% |
| Candlestick + volume (v1) | 6 / 40 | −9.7% | 3 / 40 | 44% |
| Candlestick + volume (v2) | 6 / 40 | −10.0% | 3 / 40 | 44% |
| LightGBM, next day (v1) | 0 / 40 | −28.8% | 5 / 40 | 60% |
| LightGBM, 5-day horizon | 2 / 40 | −15.3% | 2 / 40 | 52% |
| LightGBM, 20-day horizon | 5 / 40 | −9.5% | 7 / 40 | 53% |
| LightGBM, 20-day horizon (v2) | 5 / 40 | −7.5% | 6 / 40 | 55% |

Most runs sit at or near the chance level of about 2 significant stocks. The 20-day LightGBM runs are the exception: 6 to 7 of 40 is above that level before any correction for the eight runs tried (for a single run in isolation, P(≥6 of 40 | 5%) ≈ 0.014 and P(≥7 of 40 | 5%) ≈ 0.003). Even in those runs, at most 5 of 40 stocks beat buy-and-hold on CAGR and 7 of 40 on Sharpe. So beating random timing mostly meant losing less than random timing, not earning more than holding the stock.

LightGBM is trained walk-forward (500-day train, 60-day test, gap ≥ horizon). The per-stock research packets report its out-of-sample directional accuracy next to the accuracy of always predicting the majority class; in the examples examined the gap is small (for instance 51.8% vs 51.4%). The counts above are not Bonferroni-adjusted across the eight runs.

Overall, the pattern is consistent with weak-form efficiency net of costs for these stocks. Some strategies reduce drawdowns, mainly by staying out of the market, but almost none improve risk-adjusted returns.

## 8. Forward Paper Tracking

With the holdout consumed, the project records future signals for registered rules in append-only logs:
- **Frozen rules.** The rule hash is frozen; editing the file stops tracking.
- **No back-filling.** A late signal fills at the first open after it was recorded, never at a past open.
- **Same comparisons.** Every day is compared with 0050 and five matched random portfolios using fixed seeds.
- **Data gates.** A coverage gate defers a signal day when less than 95% of the price data has arrived.

No rule has yet earned enough evidence to be tracked as a candidate. The infrastructure exists so that any future pass is tested on data that did not exist when it was designed.

## 9. Limitations

- **Fundamental restatements.** Values reflect the latest FinMind version, not the original announcements.
- **Optimistic delisting.** Liquidating at the last close is optimistic for bankruptcies; a zero-recovery stress test is reported.
- **Intraday microstructure.** Only the open is checked against price limits; intraday limit locks and queue priority are not modelled.
- **Short sample, one market.** About 7.5 years of development data, dominated by the 2020 to 2021 rally, with a holdout in which large caps led (0050 returned 54.1% a year against 18.1% for the equal-weight eligible pool).
- **Survivorship in Section 7.** The 40-stock timing universe contains only currently listed firms, so those results are optimistic. The selection rules in Section 5 use the full universe including delisted stocks.
- **Power.** With 100 random portfolios and a growing N, the protocol's minimum achievable adjusted p-value rises over time; the number of random portfolios must increase as more rules are tried.

## 10. Conclusion

Under realistic Taiwan-market frictions and a control that matches holdings and turnover, none of the four selection rules, including one with a literature basis and a 28% development-period CAGR, produced risk-adjusted returns that beat a dividend-inclusive index ETF by more than luck. The single holdout observation agrees: the best development-period rule lagged the benchmark by about 42 percentage points a year.

Practically, the evidence supports a default of not adopting any rule until it survives a matched control, robustness checks and forward tracking. Methodologically, the most consequential design choice was not the model but the **baseline**: naive random portfolios made a turnover-heavy momentum rule look highly significant, and a turnover-matched control did not.

## References

- Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects. *Journal of Financial Markets*, 5(1), 31–56.
- Amihud, Y., & Mendelson, H. (1986). Asset pricing and the bid-ask spread. *Journal of Financial Economics*, 17(2), 223–249.
- Asness, C., Frazzini, A., Israel, R., Moskowitz, T., & Pedersen, L. H. (2018). Size matters, if you control your junk. *Journal of Financial Economics*, 129(3), 479–509.
- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2014). Pseudo-mathematics and financial charlatanism: the effects of backtest overfitting on out-of-sample performance. *Notices of the AMS*, 61(5), 458–471.
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). … and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68.
- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: implications for stock market efficiency. *Journal of Finance*, 48(1), 65–91.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

## Reproducibility

Every lab report embeds the configuration hash, data-snapshot hash, git commit and package versions. The rule files are in `strategies/lab/`, the full reports in `reports/lab/`, and the append-only ledger in `reports/lab/trials.csv`. Run a rule with:

```
python -m src.cli lab strategies/lab/<rule>.yaml
```
