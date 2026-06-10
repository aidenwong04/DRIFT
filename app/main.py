# This is the main entry point for the DRIFT application. 
# It sets up the FastAPI app and defines the root endpoint.

import time
import io
import numpy as np
from PIL import Image
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import onnxruntime as ort

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

CLASS_NAMES = [
    "Adobe Firefly",
    "Dall-E 3",
    "Flux.1",
    "Flux.1.1 Pro",
    "Freepik",
    "Leonardo AI",
    "Midjourney",
    "Stable Diffusion 3.5",
    "Stable Diffusion XL",
    "Starry AI",
]

@asynccontextmanager
async def lifespan(app: FastAPI): #lifespan function to manage the lifecycle of the app, including setup and teardown
    global session, probe_model, device, input_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    session = ort.InferenceSession(
        "..\\models\\drift_dinov2.onnx",
        providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    checkpoint = torch.load("..\\models\\linear_probe.pth", map_location=device)

    probe_model = nn.Linear(768,10).to(device)
    probe_model.load_state_dict(checkpoint['classifier_state'])
    probe_model.eval()

    yield  # app runs here
    # cleanup on shutdown if needed

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "DRIFT API is running"}

class PredictionResponse(BaseModel):
    prediction: str # The predicted class label
    confidence: float # The confidence score of the prediction
    latency: float # The time taken to make the prediction

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess the input image for the model."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224)) # Resize to model's expected input size
    image_array = np.array(image).astype(np.float32) / 255.0 # Normalize to [0, 1]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) # matches the mean used during model training
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_array = (image_array - mean) / std # Standardize
    image_array = np.transpose(image_array, (2, 0, 1)).astype(np.float32) # Change to (C, H, W)
    return image_array[np.newaxis, :].astype(np.float32) # Add batch dimension

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Endpoint to receive an image and return the predicted class label."""
    try:    
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)

        start_time = time.perf_counter()

        # run inference with onnxruntime, then pass the output through the probe model to get the final prediction
        features, _ = session.run(None, {input_name: input_tensor})
        features_tensor = torch.from_numpy(features).to(device)

        with torch.inference_mode():
            logits = probe_model(features_tensor) # pass through the probe model
            # normalize the output to get confidence scores
            confidence_scores = torch.softmax(logits, dim=1).cpu().numpy()[0] # convert to numpy array and get the first (and only) sample in the batch
        
        predicted_class = np.argmax(confidence_scores) # get the index of the class with the highest confidence
        confidence = float(confidence_scores[predicted_class]) # get the confidence score for the predicted class
        
        latency = (time.perf_counter() - start_time) * 1000 # calculate latency in milliseconds

        if confidence < 0.6:
            prediction = "Unknown"
        else:
            prediction = CLASS_NAMES[predicted_class]

        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            latency=round(latency, 2)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the image: {str(e)}")



    