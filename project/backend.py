from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import shutil

app = FastAPI()

# Load the model once during startup
model = tf.keras.models.load_model('potatoes.h5', compile=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define class names
class_names = ['Early Blight', 'Late Blight', 'Healthy']

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_file = f"./temp/{file.filename}"
    with open(temp_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Preprocess the image
    img = load_img(temp_file, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)
    predicted_label = class_names[predicted_class[0]]

    # Return the prediction
    return JSONResponse(content={"predicted_class": int(predicted_class[0]), "predicted_label": predicted_label})
