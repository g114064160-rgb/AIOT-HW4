# Deployment & Operations Guide

本指南說明如何在本機或伺服器上部署八哥辨識器 Streamlit 應用，前提是模型已在 Notebook 訓練完成並導出 `myna_resnet50v2.h5`。

## 環境需求
- Python 3.9+
- 作業系統：Windows / macOS / Linux 皆可（TensorFlow 安裝請依平台文件）。
- 套件：見 `requirements.txt`（`streamlit`, `tensorflow`, `Pillow`, `numpy`）。

## 安裝步驟
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 放置模型
- 將已訓練好的模型檔放在專案根目錄，預設檔名：`myna_resnet50v2.h5`。
- 若使用其他檔名或路徑，可在側欄輸入後點「載入模型」。

## 啟動應用
```bash
streamlit run app.py
```
執行後依終端顯示的本地網址（通常是 `http://localhost:8501`）開啟瀏覽器。

## 目錄結構建議
```
AIOT-HW4/
├─ app.py
├─ requirements.txt
├─ ARCHITECTURE.md
├─ DEPLOYMENT.md
├─ TROUBLESHOOTING.md
├─ proposal.md
├─ README.md
├─ myna_resnet50v2.h5          # 已訓練模型 (自行放置)
└─ crested_myna/ javan_myna/ common_myna/   # 範例圖片 (可選)
```

## 運維建議
- 確保模型與訓練時的前處理一致（224x224 + `preprocess_input`）。
- 若部署在伺服器，建議以 `nohup` 或系統服務方式常駐；Streamlit 自帶 HTTP 服務，需自行放在內部網或反向代理後面。
- 定期更新安全性修補與套件，避免長期舊版 TensorFlow 的安全風險。
