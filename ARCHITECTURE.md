# System Architecture: 八哥辨識器 (Streamlit + TensorFlow)

本系統聚焦「推論展示」，訓練仍在 Notebook 內完成。以下描述執行期組件與資料流。

## 組件
- `app.py`：Streamlit 前端與推論服務，處理圖片上傳、前處理、呼叫模型並視覺化結果。
- TensorFlow 模型：預設 `myna_resnet50v2.h5`，為 ResNet50V2 去頂後接 Dense(N, softmax) 的遷移學習模型。
- 範例圖片目錄（可選）：`./crested_myna/`, `./javan_myna/`, `./common_myna/`。若不存在，介面仍可上傳圖片。
- 設定常數：`CATEGORY_EN`, `CATEGORY_ZH`, `IMAGE_SIZE`, `DEFAULT_MODEL_PATH`（位於 `app.py`）。

## 資料流
1) 使用者上傳或選擇範例圖片。
2) 讀檔並確保 RGB → 重新縮放至 224x224 → `preprocess_input`。
3) 將張量送入載入好的模型 → 取得 softmax 機率。
4) 將 Top-1 及全類別分數在頁面顯示（長條圖 + 文字）。

## 關鍵設計決策
- **推論快取**：使用 `@st.cache_resource` 快取模型載入，避免重複載入成本。
- **尺寸與前處理**：固定 224x224 並使用 ResNet50V2 官方 `preprocess_input`，確保與訓練一致。
- **錯誤處理**：模型缺失、輸出維度不符、讀圖失敗時提供明確訊息，避免崩潰。
- **非目標**：不在 Streamlit 內訓練；不提供雲端部署腳本。

## 擴充點
- 類別清單改為可讀設定檔或 UI 表單。
- 支援載入多個模型、切換不同權重。
- 加入簡易日誌與推論統計（例如計數或延遲摘要）。
