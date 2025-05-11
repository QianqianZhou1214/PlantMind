from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

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


    return

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)