from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection import SSD300_VGG16_Weights
import cv2
from PIL import Image
import numpy as np
import io
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import time
app = FastAPI()
origins = [
    "http://localhost:8000",
    "http://192.168.1.111:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = ssd300_vgg16(  weights=SSD300_VGG16_Weights.COCO_V1)
model.eval()
COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
        'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
        'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
# Preprocessing function
def preprocess_image(image):
    preprocess = transforms.Compose([
        #transforms.Resize((504,896)),
        transforms.Resize((600,600)),
        transforms.ToTensor(),
    ])
    img_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    return img_tensor

# Visualization function (modified to return image)
def visualize_predictions(img, predictions, threshold=0.7):
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

   
    height, width = img.shape[:2]
    print("height: ",height)
    print("width: ",width)
    # img =  cv2.resize(img, (896,504))
    img =  cv2.resize(img, (600,600))

    for i, score in enumerate(scores):
        if score > threshold:
            box = boxes[i]
            print("labels[",i,"] ",labels[i])
            label = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]

            # Draw bounding box on the original image
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(img, f'{label}: {score:.2f}', (int(box[0]), int(box[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return img

# Prediction function (return list of rectangles)
def get_predictions(predictions, threshold=0.7):
    boxes = predictions[0]['boxes'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()

    
    rectangles = []

    for i, score in enumerate(scores):
        if score > threshold:
            box = boxes[i]
            label = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]

            # Format the bounding box and other info in a dictionary
            rectangles.append({
                "label": label,
                "score": float(score),
                "box": {
                    "x_min": int(box[0]),
                    "y_min": int(box[1]),
                    "x_max": int(box[2]),
                    "y_max": int(box[3])
                }
            })

    return rectangles
# FastAPI route for vehicle detection
@app.post("/detect-vehicles")
async def detect_vehicles(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))  # Load image from uploaded file

    # Convert image to RGB and then OpenCV format (for visualization)
    image = image.convert("RGB")
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # Convert RGB to BGR (OpenCV format)

    # Preprocess the image for SSD
    img_tensor = preprocess_image(image)

    # Perform inference
    with torch.no_grad():
        predictions = model(img_tensor)

    # Visualize and add bounding boxes to the image
    result_image = visualize_predictions(open_cv_image, predictions)

    print("img_encoded: 1")
    # Convert result image to bytes to send as response
    _, img_encoded = cv2.imencode('.jpg', result_image)
    
    print("img_encoded: 2")
    # return {
    #     "filename": file.filename,
    #     "prediction": img_encoded.tobytes(),
    # }
    image_stream =io.BytesIO(img_encoded)
    return StreamingResponse(content=image_stream, media_type="image/jpg")

@app.post("/getdetect")
async def detect_vehicles(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))  # Load image from uploaded file

    # Convert image to RGB
    image = image.convert("RGB")

    # Preprocess the image for SSD
    img_tensor = preprocess_image(image)
    start_time = time.time()  # Start timing
    # Perform inference
    with torch.no_grad():
        predictions = model(img_tensor)

    end_time = time.time()  # End timing
    processing_time = end_time - start_time  # Calculate elapsed time

    # Extract rectangles from predictions
    rectangles = get_predictions(predictions, threshold=0.7)

    return {"filename": file.filename, "rectangles": rectangles, "processing_time": processing_time}

