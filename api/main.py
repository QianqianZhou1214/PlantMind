from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL = tf.saved_model.load("../models/4")  # Load your model here

CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']  # Replace with your actual class names

@app.get("/hello")
async def ping():
    return "Hello, World!"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))  # Convert bytes to a file-like object, and open it with PIL as an image
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
):
    image = read_file_as_image(await file.read()) # put first thread in suspend mode so second thread can run
    img_batch = np.expand_dims(image, axis=0)  # Add batch dimension


    predictions = MODEL.predict(img_batch)  # Predict the class of the image
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])  # Get the maximum prediction value

    return {
        'clss': predicted_class,
        'confidence': float(confidence),
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)