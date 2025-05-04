import streamlit as st
import numpy as np
from PIL import Image
import requests
import io
import matplotlib.pyplot as plt

# URL API
API_URL = "https://nein.onrender.com/predict"

# Настройка страницы
st.set_page_config(page_title="Image Classifier",
                   layout="centered", page_icon="📸")

# Кастомный CSS для стилизации
st.markdown("""
    <style>
    .main {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stFileUploader {
        background-color: #2e2e2e;
        border-radius: 10px;
        padding: 10px;
    }
    .stImage {
        border: 2px solid #4CAF50;
        border-radius: 10px;
    }
    .stSpinner {
        color: #4CAF50;
    }
    h1, h2, h3 {
        color: #4CAF50;
    }
    .success-box {
        background-color: #2e7d32;
        padding: 10px;
        border-radius: 8px;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Заголовок и описание
st.title("📸 Классификатор животных")
st.markdown("Загрузите изображение 🐔🐘🐎 и узнайте, какое животное на нём!")

# Контейнер для загрузки изображения
with st.container():
    st.markdown("### Загрузите изображение")
    uploaded_file = st.file_uploader("Выберите PNG, JPG или JPEG", type=[
                                     "png", "jpg", "jpeg"], key="uploader")
    image = None
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        # Отображение изображения в колонке
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(image, caption="Загруженное изображение",
                     use_container_width=True)
        with col2:
            st.markdown("#### Готово к классификации!")

# Отправка изображения на API
if image is not None:
    st.markdown("### Классификация")
    # Предобработка
    image = image.resize((128, 128))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    if st.button("🔍 Классифицировать"):
        with st.spinner("🕒 Анализируем изображение..."):
            files = {"file": ("image.png", img_bytes, "image/png")}
            try:
                response = requests.post(API_URL, files=files)
                if response.ok:
                    result = response.json()
                    pred_class = result["predicted_class"]
                    probabilities = result["probabilities"]

                    # Emoji для классов
                    class_emoji = {"chicken": "🐔", "slon": "🐘", "horse": "🐎"}
                    st.markdown(
                        f"<div class='success-box'>Предсказанный класс: {class_emoji.get(pred_class, '❓')} <b>{pred_class}</b></div>", unsafe_allow_html=True)

                    # Визуализация вероятностей
                    st.markdown("### 📊 Вероятности классов")
                    labels = list(probabilities.keys())
                    values = list(probabilities.values())
                    # Цвета для chicken, slon, horse
                    colors = ["#FF6F61", "#6B5B95", "#88B04B"]

                    fig, ax = plt.subplots(figsize=(6, 3))
                    bars = ax.barh(labels, values, color=colors)
                    ax.set_xlabel("Вероятность")
                    ax.set_xlim(0, 1)
                    # Добавление значений на график
                    for bar, value in zip(bars, values):
                        ax.text(value, bar.get_y() + bar.get_height()/2, f"{value:.2%}",
                                va="center", ha="left", color="white", fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.error(
                        f"Ошибка API: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Не удалось подключиться к серверу: {e}")
