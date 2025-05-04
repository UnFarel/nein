from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import io
from PIL import Image

app = FastAPI(title="Image Classifier API")

# Загрузка модели
model = load_model("model/animal.keras")

# Классы
classes = {0: 'chicken', 1: 'slon', 2: 'horse'}

INPUT_SIZE = (128, 128)


def preprocess_image(file: UploadFile) -> np.ndarray:
    """Читает и преобразует изображение в формат, пригодный для модели"""
    image_bytes = file.file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(INPUT_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, 3)
    return img_array


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        img_array = preprocess_image(file)
        predictions = model.predict(img_array)
        probs = predictions[0]
        class_idx = int(np.argmax(probs))
        class_label = classes[class_idx]

        return JSONResponse({
            "predicted_class": class_label,
            "probabilities": {
                label: float(prob) for label, prob in zip(classes.values(), probs)
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
