import io
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import streamlit as st
from PIL import Image
import requests

# TensorFlow 依賴：若缺少會在 UI 顯示清楚錯誤
TF_AVAILABLE = False
try:
    import tensorflow as tf  # noqa
    TF_AVAILABLE = True
except ModuleNotFoundError:
    TF_AVAILABLE = False


# 基本設定
CATEGORY_EN = ["crested_myna", "javan_myna", "common_myna"]
CATEGORY_ZH = ["土八哥", "白尾八哥", "家八哥"]
DEFAULT_MODEL_PATH = "assets/myna_logreg.npz"
IMAGE_SIZE = (224, 224)

# 內建範例（含八哥與非八哥），使用本地檔避免外部連線問題
SAMPLE_IMAGES = {
    "八哥-白尾": "assets/samples/javan_myna.jpg",
    "八哥-家八哥": "assets/samples/common_myna.jpg",
    "八哥-土八哥": "assets/samples/crested_myna.jpg",
    "非八哥-鳥1": "assets/samples/non_myna_bird1.jpg",
    "非八哥-鳥2": "assets/samples/non_myna_bird2.jpg",
}


def load_image(image_file: Union[Path, str, io.BytesIO]) -> Image.Image:
    """讀入影像並轉成 RGB，支援本地路徑、URL、記憶體緩衝。"""
    if isinstance(image_file, (str, Path)):
        s = str(image_file)
        if s.startswith("http://") or s.startswith("https://"):
            resp = requests.get(
                s,
                timeout=15,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0)",
                    "Accept": "image/*,*/*;q=0.8",
                },
            )
            resp.raise_for_status()
            buf = io.BytesIO(resp.content)
            img = Image.open(buf)
        else:
            img = Image.open(s)
    else:
        img = Image.open(image_file)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


@st.cache_resource(show_spinner=False)
def load_tf_model(model_path: str):
    """載入 TensorFlow 模型，並在 Streamlit 端做快取。"""
    from tensorflow.keras.models import load_model
    return load_model(model_path)


@st.cache_resource(show_spinner=False)
def load_logreg_model(model_path: str):
    """載入輕量 logistic regression 模型 (numpy 儲存)。"""
    data = np.load(model_path)
    return {
        "w": data["w"],
        "b": data["b"],
        "mean": data["mean"],
        "std": data["std"],
    }


def preprocess_logreg(img: Image.Image, target_size=(64, 64)) -> np.ndarray:
    """調整尺寸、轉為向量，提供給輕量化 softmax 模型。"""
    img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
    arr = np.array(img_resized).astype(np.float32) / 255.0  # (H,W,3)
    return arr.reshape(1, -1)  # (1, D)


def predict_logreg(model_params: dict, img: Image.Image, labels: List[str]) -> Tuple[str, float, List[float]]:
    """使用預先訓練好的 logistic regression (numpy) 進行推論。"""
    w = model_params["w"]
    b = model_params["b"]
    mean = model_params["mean"]
    std = model_params["std"]

    x = preprocess_logreg(img)
    x = (x - mean) / (std + 1e-6)
    logits = x @ w + b
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    probs = (exp / exp.sum(axis=1, keepdims=True)).flatten()
    if len(probs) != len(labels):
        raise ValueError(f"模型輸出維度 ({len(probs)}) 與標籤數 ({len(labels)}) 不符")
    top_idx = int(np.argmax(probs))
    return labels[top_idx], float(probs[top_idx]), probs.tolist()


def discover_sample_images(base_dir: Path, categories: List[str]) -> List[Path]:
    """嘗試尋找範例圖片；若資料夾不存在則回傳空清單。"""
    samples: List[Path] = []
    for cat in categories:
        cat_dir = base_dir / cat
        if not cat_dir.exists():
            continue
        for fname in os.listdir(cat_dir):
            path = cat_dir / fname
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                samples.append(path)
    return samples


def main():
    st.set_page_config(page_title="八哥辨識器", page_icon="🐦", layout="wide")
    st.title("八哥辨識器 (輕量版, logistic regression)")
    st.markdown("上傳或選擇範例圖片，使用內建輕量模型辨識三類八哥。")

    # Sidebar: 模型與輸入
    st.sidebar.header("設定")
    model_path = st.sidebar.text_input("模型路徑", value=DEFAULT_MODEL_PATH)
    load_model_btn = st.sidebar.button("載入模型")

    uploaded = st.sidebar.file_uploader("上傳圖片", type=["jpg", "jpeg", "png", "bmp", "webp"])

    sample_images = discover_sample_images(Path("."), CATEGORY_EN)
    sample_options = ["(不使用範例)"] + list(SAMPLE_IMAGES.keys())
    if sample_images:
        sample_options += [f"(本地){p}" for p in sample_images]
    sample_choice: Optional[str] = st.sidebar.selectbox("快速範例", options=sample_options)

    # 主體區
    col1, col2 = st.columns([1, 1])
    image: Optional[Image.Image] = None
    image_name: Optional[str] = None

    # 讀取圖片
    if uploaded is not None:
        try:
            image = load_image(uploaded)
            image_name = uploaded.name
        except Exception as e:
            st.error(f"讀取上傳圖片失敗：{e}")
    elif sample_choice and sample_choice != "(不使用範例)":
        try:
            if sample_choice in SAMPLE_IMAGES:
                image = load_image(SAMPLE_IMAGES[sample_choice])
                image_name = sample_choice
            elif sample_choice.startswith("(本地)"):
                p = Path(sample_choice.replace("(本地)", "", 1))
                image = load_image(p)
                image_name = p.name
            else:
                image = None
        except Exception as e:
            st.error(f"讀取範例圖片失敗：{e}")

    with col1:
        st.subheader("輸入圖片")
        if image is not None:
            st.image(image, caption=image_name or "輸入圖片", use_column_width=True)
        else:
            st.info("請上傳圖片或選擇範例。")

    # 載入模型
    model_logreg = None
    model_error = None
    if load_model_btn or Path(model_path).exists():
        if not model_path or not Path(model_path).exists():
            model_error = f"找不到模型檔案：{model_path}"
        else:
            try:
                with st.spinner("載入模型中..."):
                    model_logreg = load_logreg_model(model_path)
            except Exception as e:
                model_error = f"模型載入失敗：{e}"

    if model_error:
        st.error(model_error)

    # 推論
    with col2:
        st.subheader("推論結果")
        if image is not None and model_logreg is not None:
            if st.button("開始辨識", type="primary"):
                try:
                    top_label, top_score, scores = predict_logreg(model_logreg, image, CATEGORY_ZH)
                    st.success(f"Top-1: {top_label} ({top_score:.2%})")
                    chart_data = {
                        "label": CATEGORY_ZH,
                        "probability": scores,
                    }
                    st.bar_chart(chart_data, x="label", y="probability", use_container_width=True)
                except Exception as e:
                    st.error(f"推論失敗：{e}")
        elif image is None:
            st.info("尚未選擇圖片。")
        elif model_logreg is None:
            st.info("請先載入模型。")

    # 範例圖片提示
    if not sample_images:
        st.caption("未找到本地範例資料夾，已提供線上範例（含八哥與非八哥）。")


if __name__ == "__main__":
    main()
