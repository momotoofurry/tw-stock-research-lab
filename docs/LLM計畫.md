# 本機 LLM 報告與 Edge AI 實驗計畫

個人研究與 Edge AI 課程用。整合進本專案，沿用既有的資料、回測、研究資料包與介面。

## 原則（和專案規則一致）
- LLM 不預測股價、不給買賣指令；只把程式算好的數字寫成白話
- 所有數字由 Python 計算；LLM 輸出用**數字查核器**逐一比對輸入，對不上的標出來（也是品質指標「數字忠實率」）
- 報告標示資料日期與來源，結尾「本報告僅供個人研究參考，非投資建議」
- 全部本機執行，不用雲端 LLM API；學校伺服器只放公開股票資料，不放 FinMind token、`portfolio.yaml`

## 設備
| | 筆電 | 學校 GPU 伺服器（2 × T4）（可用到 2026-12-31） |
|---|---|---|
| GPU | RTX 3060 筆電版 6GB（桌面程式約佔 0.6GB） | 2 × Tesla T4 16GB（無 bf16；GPU 0 使用者作業在用，先只用 GPU 1） |
| CPU / RAM | i9-12900H / 約 40GB | 8 核 / 62GB（作業執行中約剩 18GB） |
| 系統 | Windows 11，`.venv` Python 3.12 | Ubuntu 22.04 容器，Python 3.10，CUDA 12.2 驅動 |

## 模型：Qwen3（Apache-2.0，原始權重 BF16）
- 主力：Qwen3-4B-Instruct-2507（只有非思考模式，適合寫報告）
- 對照：Qwen3-1.7B（蒸餾學生 / 更小的量化對照）、Qwen3-8B（offload 實驗）
- 非思考模式建議取樣：temperature 0.7、top_p 0.8、top_k 20、min_p 0；量測時固定 seed
- T4 不支援 bf16，FP16 基準要先把 BF16 權重轉成 FP16

## 階段
1. **環境**：筆電裝 Ollama，確認模型跑在 GPU 上
2. **報告產生器**：讀 `python -m src.cli research <代號>` 的資料包 → 本機 LLM 寫白話報告 → 數字查核器
3. **量化實驗**：自己從 FP16 做 Q8 / Q5 / Q4 / Q3；通用 vs 台股報告的 imatrix 校準；KV cache 量化；量 TTFT、tokens/sec、VRAM、數字忠實率；筆電 vs T4（頻寬相近、算力不同 → memory-bound 對照）
4. **蒸餾**（T4）：老師模型讀資料包寫報告 → 查核器過濾 → 微調 1.5B～4B 學生 → 量化 → 筆電部署
5. **回測解說**：接現有回測結果
6. **RAG 與介面整合**（最後做）

時程：GPU 吃重的部分（老師產生資料、蒸餾、FP16 基準）排在 12 月底前。

## 量測注意
- 固定資料快照（同一批資料包）、固定 prompt 與 seed
- 每組先暖機再重複量測；筆電固定電源模式並注意降頻
- TTFT、tokens/sec 用 Ollama 回應裡的時間欄位；VRAM 用 `nvidia-smi`
