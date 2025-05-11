from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"  # Replace with your actual endpoint


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

    json_data = {
        "instances": img_batch.tolist()
    }

    requests.post(endpoint, json=json_data)
    
    pass
  

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)