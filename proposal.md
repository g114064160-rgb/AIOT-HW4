# Proposal: Streamlit 八哥辨識器 (ResNet50V2 遷移學習)

## 目標
以 Notebook「【Demo02】遷移式學習做八哥辨識器.ipynb」為基礎，將推論與展示功能封裝成 Streamlit 應用（`app.py`），便於上傳圖片即時辨識，並可載入已訓練好的模型權重。

## 功能需求
- 圖片上傳與範例選擇：支援使用者上傳圖片，並提供數張內建示例（可由資料夾掃描或固定 URL）。
- 模型載入：優先從本地檔案（預設 `myna_resnet50v2.h5`）載入；如不存在，提示使用者先在 Notebook 訓練並輸出模型。
- 推論流程：
  - 將圖片轉為 RGB，縮放至 224x224。
  - 使用 `tensorflow.keras.applications.resnet_v2.preprocess_input` 做前處理。
  - 呼叫載入模型進行預測，輸出每一類的機率，並標示 Top-1 結果。
- 類別設定：從常數清單載入（預設土八哥/白尾八哥/家八哥），若模型輸出維度不符則顯示錯誤提示。
- UI 呈現：
  - 左側：上傳區／範例下拉；模型路徑輸入框；推論按鈕。
  - 右側：顯示原圖、Top-1 結果與信心度、各類別長條圖。
- 錯誤處理：
  - 模型檔缺失／載入失敗時顯示訊息。
  - 非影像或讀檔失敗時顯示提示。

## 非目標
- 不在 Streamlit 內訓練模型；訓練仍由 Notebook 進行。
- 不實作雲端部署腳本，僅產出本地可執行的 `app.py`。

## 輸出物
1) `proposal.md`（本文件）。
2) `app.py`：Streamlit 前端 + 推論邏輯，依賴 `tensorflow`, `Pillow`, `numpy`, `streamlit`, `matplotlib` 或 `plotly`（使用哪個以程式為準）。

## 使用方式（預期）
```bash
pip install -r requirements.txt  # 若無此檔，可手動安裝 streamlit tensorflow pillow numpy
streamlit run app.py
```
在執行前請先放置已訓練好的模型檔案（預設 `myna_resnet50v2.h5`）至專案目錄，或於介面中指定路徑。
