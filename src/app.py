import json

import av
import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    webrtc_streamer,
)

from config import (
    CLASS_INDICES_PATH,
    CNN_MODEL_PATH,
    EFF_MODEL_PATH,
    IMG_SIZE,
    RESNET_MODEL_PATH,
    TEST_DIR,
)

from recommendation import get_recommendation
# from PIL import ImageFont
# from PIL import ImageDraw

st.set_page_config(
    page_title="Waste Classification",
    layout="wide"
)

# ================= CSS =================
st.markdown(
    """
    <style>

    .stApp {
        background-color: #f5f7fa;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ================= TITLE =================
st.title("♻️ Waste Material Classification System")

st.caption(
    "Ensemble AI đang sử dụng EfficientNetB0 + ResNet50"
)

st.sidebar.title("📌 About")

st.sidebar.info(
    """
    Waste Material Classification System

    - EfficientNetB0
    - ResNet50
    - Ensemble AI
    - Realtime Detection
    - Streamlit Interface
    """
)

# ================= MODEL =================
MODEL_OPTIONS = {
    "EfficientNetB0": {
        "path": EFF_MODEL_PATH,
        "type": "effnet"
    },

    "ResNet50": {
        "path": RESNET_MODEL_PATH,
        "type": "resnet"
    },

    "CNN": {
        "path": CNN_MODEL_PATH,
        "type": "cnn"
    }
}


# ================= LOAD =================
@st.cache_data(show_spinner=False)
def load_class_names():

    with open(
        CLASS_INDICES_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        class_indices = json.load(f)

    inv = {
        v: k
        for k, v in class_indices.items()
    }

    class_names = [
        inv[i]
        for i in range(len(inv))
    ]

    return class_names


@st.cache_resource(show_spinner=False)
def load_model(model_path):

    return tf.keras.models.load_model(
        model_path,
        compile=False
    )


@st.cache_resource(show_spinner=False)
def load_assets():

    eff_model = load_model(EFF_MODEL_PATH)

    res_model = load_model(RESNET_MODEL_PATH)

    class_names = load_class_names()

    return eff_model, res_model, class_names


eff_model, res_model, class_names = load_assets()

# ================= TIENG VIET =================
VIETNAMESE_LABELS = {
    "Alu": "Lon nhom",
    "Carton": "Bia carton",
    "Foam_box": "Hop xop",
    "Milk_box": "Hop sua",
    "Paper": "Giay",
    "Paper_cup": "Coc giay",
    "PET": "Chai nhua PET",
    "Plastic_cup": "Coc nhua",
    "Other": "Rac khac"
}

# ================= PREPROCESS =================
def preprocess_image(image, model_type):

    img = image.convert("RGB").resize(IMG_SIZE)

    x = np.array(
        img,
        dtype=np.float32
    )

    x = np.expand_dims(x, axis=0)

    if model_type == "cnn":

        x = x / 255.0

    elif model_type == "effnet":

        x = tf.keras.applications.efficientnet.preprocess_input(x)

    elif model_type == "resnet":

        x = tf.keras.applications.resnet50.preprocess_input(x)

    return x


def predict_probs(model, x):

    return model.predict(
        x,
        verbose=0
    )[0]


# ================= ENSEMBLE =================
def ensemble_predict(image):

    x_eff = preprocess_image(
        image,
        "effnet"
    )

    x_res = preprocess_image(
        image,
        "resnet"
    )

    prob_eff = predict_probs(
        eff_model,
        x_eff
    )

    prob_res = predict_probs(
        res_model,
        x_res
    )

    # tăng trọng số EfficientNet
    prob = (
        0.75 * prob_eff +
        0.25 * prob_res
    )

    return (
        prob,
        class_names
    )

# ================= REALTIME =================
RTC_CONFIGURATION = {
    "iceServers": [
        {
            "urls": [
                "stun:stun.l.google.com:19302"
            ]
        }
    ]
}


class VideoProcessor(VideoProcessorBase):

    def __init__(self):

        self.eff_model = eff_model
        self.res_model = res_model
        self.class_names = class_names

        self.frame_count = 0

        self.last_text = "..."

        # prediction history
        self.pred_history = []

        # giữ prediction hiện tại
        self.current_class = ""

        # cooldown chống đứng prediction
        self.same_count = 0

    def recv(self, frame):

        img = frame.to_ndarray(format="bgr24")

        self.frame_count += 1

        # predict 
        if self.frame_count % 6 == 0:

            # resize realtime
            h, w, _ = img.shape

            crop = img[
                h//4:h*3//4,
                w//4:w*3//4
            ]

            small = cv2.resize(
                crop,
                (160, 160)
            )

            rgb = cv2.cvtColor(
                small,
                cv2.COLOR_BGR2RGB
            )

            pil_img = Image.fromarray(rgb)

            # preprocess
            x_eff = preprocess_image(
                pil_img,
                "effnet"
            )

            x_res = preprocess_image(
                pil_img,
                "resnet"
            )

            # predict
            # prob_eff = predict_probs(
            #     self.eff_model,
            #     x_eff
            # )

            # prob_res = predict_probs(
            #     self.res_model,
            #     x_res
            # )

            # # ensemble
            # final_prob = (
            #     0.8 * prob_eff +
            #     0.2 * prob_res
            # )
            final_prob = predict_probs(
                self.eff_model,
                x_eff
            )

            pred_idx = int(
                np.argmax(final_prob)
            )

            pred_class_en = self.class_names[pred_idx]

            pred_class = VIETNAMESE_LABELS.get(
                pred_class_en,
                pred_class_en
            )

            conf = final_prob[pred_idx] * 100

            # ================= FILTER =================
            # confidence thấp -> detecting
            if conf < 50:

                if self.last_text == "Doi":
                    self.last_text = "Dang detect..."

            else:

                # prediction history
                self.pred_history.append(
                    pred_class
                )

                # giữ 6 prediction gần nhất
                self.pred_history = (
                    self.pred_history[-3:]
                )

                # majority voting
                final_class = max(
                    set(self.pred_history),
                    key=self.pred_history.count
                )
                final_class_vn = VIETNAMESE_LABELS.get(
                    final_class,
                    final_class
                )

                # ================= CHANGE DETECTION =================
                # nếu object đổi thật -> update nhanh
                if final_class != self.current_class:

                    self.current_class = final_class
                    self.same_count = 0

                else:
                    self.same_count += 1

                 # ===== TOP 3 =====
                top3_idx = np.argsort(
                    final_prob
                )[-3:][::-1]

                display_lines = []

                for idx in top3_idx:

                    vn_name = VIETNAMESE_LABELS.get(
                        self.class_names[idx],
                        self.class_names[idx]
                    )

                    score = (
                        final_prob[idx] * 100
                    )

                    display_lines.append(
                        f"{vn_name}: {score:.1f}%"
                    )

                self.last_text = display_lines

        # ================= UI =================

        # background box
        box_height = 40 + (
            len(self.last_text) * 30
        )

        cv2.rectangle(
            img,
            (10, 10),
            (420, box_height),
            (40, 40, 40),
            -1
        )

        # hiện text dọc
        for i, text in enumerate(self.last_text):

            cv2.putText(
                img,
                text,
                (20, 40 + i * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

# ================= UI =================
def show_probability_chart(
    prob,
    class_names
):

    df = pd.DataFrame({
        "Class": class_names,
        "Probability": [
            float(p) * 100
            for p in prob
        ]
    })
    df["Class"] = df["Class"].map(
        lambda x: VIETNAMESE_LABELS.get(x, x)
    )

    df["Class"] = df["Class"].map(
    lambda x: VIETNAMESE_LABELS.get(x, x)
)

    chart = (
        alt.Chart(df)
        .mark_bar(
            cornerRadiusTopRight=8,
            cornerRadiusBottomRight=8
        )
        .encode(
            x=alt.X(
                "Probability:Q",
                title="Xác suất (%)"
            ),

            y=alt.Y(
                "Class:N",
                sort="-x"
            ),

            tooltip=[
                "Class",
                "Probability"
            ]
        )
        .properties(
            height=250
        )
    )

    st.altair_chart(
        chart,
        use_container_width=True,
        theme=None
    )


# ================= LAYOUT =================
left_col, right_col = st.columns([1, 1.4])

with left_col:

    uploaded_file = st.file_uploader(
        "Upload ảnh",
        type=["jpg", "png", "jpeg"]
    )

    camera_img = st.camera_input(
        "Hoặc chụp ảnh"
    )

    st.divider()

    st.subheader(
        "Realtime AI Detection"
    )

    # webrtc_streamer(
    #     key="waste-ai",

    #     video_processor_factory=VideoProcessor,

    #     rtc_configuration=RTC_CONFIGURATION,

    #     media_stream_constraints={
    #         "video": True,
    #         "audio": False
    #     },

    #     desired_playing_state=True,

    #     video_html_attrs={
    #         "style": {
    #             "width": "100%",
    #             "border-radius": "10px",
    #         },
    #         "autoPlay": True,
    #         "controls": False,
    #     }
    # )
    # webrtc_streamer(
    #     key="waste-ai",

    #     video_processor_factory=VideoProcessor,

    #     rtc_configuration=RTC_CONFIGURATION,

    #     media_stream_constraints={
    #         "video": True,
    #         "audio": False
    #     },

    #     async_processing=True
    # )

    ctx = webrtc_streamer(
    key="waste-ai",

    video_processor_factory=VideoProcessor,

    rtc_configuration=RTC_CONFIGURATION,

    media_stream_constraints={
        "video": True,
        "audio": False
    },

    async_processing=True,

    video_html_attrs={
        "style": {
            "width": "100%",
            "height": "400px",
        },
        "autoPlay": True,
        "controls": False,
    }
)

    predict_btn = st.button(
        "Dự đoán"
    )


with right_col:

    st.subheader("Kết quả")

    image = None

    if uploaded_file:

        image = Image.open(uploaded_file)

    elif camera_img:

        image = Image.open(camera_img)

    if image and predict_btn:

        with st.spinner("AI đang phân tích ảnh..."):

            (
                prob,
                class_names
            ) = ensemble_predict(image)

        st.image(
            image,
            width=250
        )

        pred_idx = int(np.argmax(prob))

        pred_class_en = class_names[pred_idx]

        pred_class = VIETNAMESE_LABELS.get(
            pred_class_en,
            pred_class_en
        )

        pred_conf = prob[pred_idx] * 100

        # top3_idx = np.argsort(prob)[-3:][::-1]

        # st.markdown("### Top 3 dự đoán")

        # for idx in top3_idx:

        #     vn_name = VIETNAMESE_LABELS.get(
        #         class_names[idx],
        #         class_names[idx]
        #     )

        #     st.write(
        #         f"- {vn_name}: {prob[idx]*100:.2f}%"
        #     )

        # ================= RESULT =================
        if pred_conf >= 70:

            st.success(
                f"🎯 Dự đoán: {pred_class}"
            )

        elif pred_conf >= 50:

            st.warning(
                f"⚠️ Dự đoán: {pred_class}"
            )

        else:

            st.error(
                f"❌ Dự đoán chưa chắc chắn"
            )

        st.metric(
            label="Độ tin cậy",
            value=f"{pred_conf:.2f}%"
        )

        st.divider()

        # ================= RECOMMEND =================
        rec = get_recommendation(
            pred_class
        )

        st.markdown(
            "### Mô tả vật liệu"
        )

        st.write(
            f"**Vật liệu:** {rec['material']}"
        )

        st.write(
            f"**Nơi bỏ:** {rec['bin']}"
        )

        st.write(
            f"**Lưu ý:** {rec['advice']}"
        )

        st.info(
            f"♻️ Khả năng tái chế: {rec['recycle']}"
        )

        st.divider()

        # ================= PROBABILITY =================
        st.markdown(
            "### Xác suất từng lớp"
        )

        show_probability_chart(
            prob,
            class_names
        )

    else:

        st.info(
            "Upload ảnh hoặc chụp ảnh"
        )