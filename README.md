# 八哥辨識器：ResNet50V2 遷移學習 + Gradio Web App

使用「【Demo02】遷移式學習做八哥辨識器.ipynb」延伸的完整說明，協助你快速換資料、重新訓練並上線。

## Streamlit
- https://aiot-hw4-ai9kq6knrsnc9nk24cgarr.streamlit.app/

## 功能特色
- 遷移學習：`ResNet50V2(include_top=False, pooling="avg")`，最後接 `Dense(num_classes, softmax)`。
- 預設三類八哥：`crested_myna`(土八哥)、`javan_myna`(白尾八哥)、`common_myna`(家八哥)；可自行增減。
- 內建資料下載 (`myna.zip`) 示範流程，也能替換成自己的資料夾。
- 包含預處理、訓練、評估、推論，以及 Gradio Web 介面與範例圖片。

## 環境需求
- Python 3.9+（Colab 可直接執行）
- 套件：`tensorflow`、`numpy`、`pandas`、`matplotlib`、`Pillow`、`gradio`
- 安裝：`pip install tensorflow gradio numpy pandas matplotlib pillow`

## 快速開始（Colab）
1. 開啟 Notebook 並「在雲端硬碟中儲存副本」。
2. 調整前置參數：
   - `category_en = "crested_myna,javan_myna,common_myna"`
   - `category_zh = "土八哥,白尾八哥,家八哥"`
   - 自訂 `title`, `description`（Gradio 用）。
3. 安裝套件：`pip install gradio`（Colab 已內建 TF）。
4. 下載並解壓資料：`wget ... myna.zip` → `zipfile.ZipFile(...).extractall('/content')`。
5. 資料讀取與預處理：
   - `load_img(..., target_size=(224,224))` 讀圖；`img_to_array` 轉陣列。
   - `preprocess_input` 做 ResNet 規範化。
   - `to_categorical(target, N)` 產生 one-hot 標籤。
6. 建模與編譯：
   ```python
   resnet = ResNet50V2(include_top=False, pooling="avg")
   model = Sequential([resnet, Dense(N, activation="softmax")])
   resnet.trainable = False
   model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
   ```
7. 訓練與評估：
   - `model.fit(x_train, y_train, batch_size=10, epochs=10)`
   - `model.evaluate(x_train, y_train)` 查看 loss/accuracy。
8. Gradio 部署：
   ```python
   def classify_image(inp):
       img = Image.fromarray(inp).resize((224,224), Image.Resampling.LANCZOS)
       arr = preprocess_input(np.array(img)[None, ...])
       pred = model.predict(arr).flatten()
       return {labels[i]: float(pred[i]) for i in range(N)}

   gr.Interface(
       fn=classify_image,
       inputs=gr.Image(label="八哥照片"),
       outputs=gr.Label(num_top_classes=N, label="AI辨識結果"),
       title=title,
       description=description,
       examples=sample_images
   ).launch(debug=True, share=True)
   ```
   `sample_images` 由資料夾自動蒐集；可替換成自訂清單。

## 自訂與延伸
- 增減類別：改 `category_en` / `category_zh`，並在對應資料夾放圖片；`N` 會自動對應。
- 換資料集：替換下載與資料夾路徑，保持 `target_size=(224,224)`；若尺寸不同請同步修改 `resize_image`。
- 調整訓練：修改 `batch_size`、`epochs`，加入驗證集或資料增強 (ImageDataGenerator / tf.data)。
- 解凍微調：將 `resnet.trainable = True`，可選擇只解凍後段層並降低學習率。
- 模型儲存：`model.save("myna_resnet50v2.h5")`，推論時用 `load_model` 後呼叫 `classify_image`。
- CLI 推論：可新增腳本 `infer.py --image path.jpg --model myna_resnet50v2.h5`，重用 `classify_image` 邏輯。

## 常見問題
- 若圖片非 RGB，請先轉成 RGB 再推入模型。
- 自有資料時，每類至少數十張圖較穩定；不足可透過資料增強補強。
- `share=True` 會生成臨時公開連結；正式部署請自架或用 Hugging Face Spaces。
