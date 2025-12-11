import io
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import streamlit as st
from PIL import Image
import requests

# TensorFlow 依賴：若缺少會在 UI 顯示清楚錯誤
try:
    from tensorflow.keras.applications.resnet_v2 import preprocess_input
    from tensorflow.keras.models import load_model
except ModuleNotFoundError as e:
    st.error(
        "TensorFlow 未安裝或版本不符，請先安裝 `tensorflow` 或 `tensorflow-cpu`。"
        " 若在 Streamlit Cloud，請確認 requirements.txt 已更新並重新部署。"
        f"\n\n詳細：{e}"
    )
    st.stop()


# 基本設定
CATEGORY_EN = ["crested_myna", "javan_myna", "common_myna"]
CATEGORY_ZH = ["土八哥", "白尾八哥", "家八哥"]
DEFAULT_MODEL_PATH = "myna_resnet50v2.h5"
IMAGE_SIZE = (224, 224)

# 內建範例（含八哥與非八哥）
SAMPLE_IMAGES = {
    "八哥-白尾": "https://upload.wikimedia.org/wikipedia/commons/6/6b/Javan_Myna_Singapore.jpg",
    "八哥-家八哥": "https://upload.wikimedia.org/wikipedia/commons/7/7d/Common_Myna_%28Acridotheres_tristis%29_Photograph_by_Shantanu_Kuveskar.jpg",
    "八哥-土八哥": "https://upload.wikimedia.org/wikipedia/commons/1/16/Crested_Myna_2018-03-03.jpg",
    "非八哥-藍鵲": "https://upload.wikimedia.org/wikipedia/commons/4/40/Taiwan_Blue_Magpie.jpg",
    "非八哥-麻雀": "https://upload.wikimedia.org/wikipedia/commons/0/0c/Tree_sparrow_3.jpg",
}


def load_image(image_file: Union[Path, str, io.BytesIO]) -> Image.Image:
    """讀入影像並轉成 RGB，支援本地路徑、URL、記憶體緩衝。"""
    if isinstance(image_file, (str, Path)):
        s = str(image_file)
        if s.startswith("http://") or s.startswith("https://"):
            resp = requests.get(s, timeout=10)
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
    return load_model(model_path)


def preprocess(img: Image.Image) -> np.ndarray:
    """調整尺寸、轉成張量、套用 ResNet50V2 前處理。"""
    img_resized = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    arr = np.array(img_resized)
    arr = arr[None, ...]  # (1, 224, 224, 3)
    arr = preprocess_input(arr)
    return arr


def predict(model, img: Image.Image, labels: List[str]) -> Tuple[str, float, List[float]]:
    """跑推論，回傳 top-1 與全類別分數。"""
    arr = preprocess(img)
    preds = model.predict(arr).flatten().tolist()
    if len(preds) != len(labels):
        raise ValueError(f"模型輸出維度 ({len(preds)}) 與標籤數 ({len(labels)}) 不符")
    top_idx = int(np.argmax(preds))
    return labels[top_idx], float(preds[top_idx]), preds


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
    st.title("八哥辨識器 (ResNet50V2 遷移學習)")
    st.markdown(
        "上傳或選擇範例圖片，載入已訓練好的模型（預設 `myna_resnet50v2.h5`）後進行辨識。"
    )

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
    model = None
    model_error = None
    if load_model_btn:
        if not model_path or not Path(model_path).exists():
            model_error = f"找不到模型檔案：{model_path}"
        else:
            try:
                with st.spinner("載入模型中..."):
                    model = load_tf_model(model_path)
            except Exception as e:
                model_error = f"模型載入失敗：{e}"

    if model_error:
        st.error(model_error)

    # 推論
    with col2:
        st.subheader("推論結果")
        if image is not None and model is not None:
            if st.button("開始辨識", type="primary"):
                try:
                    top_label, top_score, scores = predict(model, image, CATEGORY_ZH)
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
        elif model is None:
            st.info("請先載入模型。")

    # 範例圖片提示
    if not sample_images:
        st.caption("未找到本地範例資料夾，已提供線上範例（含八哥與非八哥）。")


if __name__ == "__main__":
    main()
