import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "efficientnet_waste.keras"
CLASS_INDICES_PATH = ROOT / "models" / "class_indices.json"
IMG_SIZE = (224, 224)

VIETNAMESE_LABELS = {
    "Alu": "Lon nhôm",
    "Carton": "Bìa carton",
    "Foam_box": "Hộp xốp",
    "Milk_box": "Hộp sữa",
    "Paper": "Giấy",
    "Paper_cup": "Cốc giấy",
    "PET": "Chai nhựa PET",
    "Plastic_cup": "Cốc nhựa",
    "Other": "Rác khác",
}

RECOMMENDATIONS = {
    "Lon nhôm": ("Thùng tái chế kim loại", "Làm sạch và ép dẹt trước khi bỏ."),
    "Bìa carton": ("Thùng giấy tái chế", "Giữ khô, tháo băng keo và gấp gọn."),
    "Hộp xốp": ("Rác khó tái chế", "Làm sạch và kiểm tra điểm thu gom xốp tại địa phương."),
    "Hộp sữa": ("Điểm thu gom bao bì giấy nhiều lớp", "Rửa sạch, để khô và ép phẳng."),
    "Giấy": ("Thùng giấy tái chế", "Không bỏ giấy ướt hoặc dính dầu mỡ."),
    "Cốc giấy": ("Rác khó tái chế", "Tách nắp và ống hút; kiểm tra quy định thu gom địa phương."),
    "Chai nhựa PET": ("Thùng nhựa tái chế", "Rửa sạch, tháo nắp và ép gọn."),
    "Cốc nhựa": ("Thùng nhựa tái chế", "Rửa sạch và tách nắp nếu khác vật liệu."),
    "Rác khác": ("Thùng rác thông thường", "Phân loại lại nếu có thể trước khi bỏ."),
}


st.set_page_config(page_title="Waste Classification", page_icon="♻️", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #f5f7fa; }
    div[data-testid="stMetric"] {
        background: white; padding: 16px; border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with CLASS_INDICES_PATH.open("r", encoding="utf-8") as file:
        class_indices = json.load(file)
    inverse = {value: key for key, value in class_indices.items()}
    return model, [inverse[index] for index in range(len(inverse))]


def predict(image: Image.Image):
    resized = image.convert("RGB").resize(IMG_SIZE)
    values = np.asarray(resized, dtype=np.float32)
    values = np.expand_dims(values, axis=0)
    values = tf.keras.applications.efficientnet.preprocess_input(values)
    return model.predict(values, verbose=0)[0]


model, class_names = load_assets()

st.title("♻️ Hệ thống phân loại vật liệu rác thải")
st.caption("Demo sử dụng EfficientNetB0 để nhận diện 9 nhóm vật liệu từ ảnh.")

with st.sidebar:
    st.header("Giới thiệu")
    st.write(
        "Đồ án Khoa học Dữ liệu: tải ảnh hoặc chụp ảnh, nhận kết quả phân loại "
        "và gợi ý xử lý phù hợp."
    )
    st.info("Kết quả chỉ mang tính minh họa cho mục đích học tập.")

input_col, result_col = st.columns([1, 1.25], gap="large")

with input_col:
    st.subheader("1. Chọn ảnh")
    source = st.radio("Nguồn ảnh", ["Tải ảnh", "Chụp ảnh"], horizontal=True)
    if source == "Tải ảnh":
        selected = st.file_uploader("JPG, JPEG hoặc PNG", type=["jpg", "jpeg", "png"])
    else:
        selected = st.camera_input("Chụp vật liệu cần phân loại")

with result_col:
    st.subheader("2. Kết quả")
    if selected is None:
        st.info("Hãy chọn một ảnh để bắt đầu phân loại.")
    else:
        image = Image.open(selected)
        st.image(image, caption="Ảnh đầu vào", width=320)
        if st.button("Phân loại", type="primary", use_container_width=True):
            with st.spinner("Đang phân tích ảnh..."):
                probabilities = predict(image)

            predicted_index = int(np.argmax(probabilities))
            english_label = class_names[predicted_index]
            vietnamese_label = VIETNAMESE_LABELS.get(english_label, english_label)
            confidence = float(probabilities[predicted_index]) * 100

            st.success(f"Kết quả: **{vietnamese_label}**")
            st.metric("Độ tin cậy", f"{confidence:.2f}%")

            bin_name, advice = RECOMMENDATIONS.get(
                vietnamese_label,
                ("Kiểm tra quy định địa phương", "Phân loại và làm sạch trước khi bỏ."),
            )
            st.write(f"**Nơi bỏ đề xuất:** {bin_name}")
            st.write(f"**Lưu ý:** {advice}")

            chart_data = pd.DataFrame(
                {
                    "Nhóm": [VIETNAMESE_LABELS.get(name, name) for name in class_names],
                    "Xác suất": [float(value) * 100 for value in probabilities],
                }
            )
            chart = (
                alt.Chart(chart_data)
                .mark_bar(cornerRadiusEnd=5)
                .encode(
                    x=alt.X("Xác suất:Q", title="Xác suất (%)"),
                    y=alt.Y("Nhóm:N", sort="-x", title=None),
                    tooltip=[alt.Tooltip("Nhóm:N"), alt.Tooltip("Xác suất:Q", format=".2f")],
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)

