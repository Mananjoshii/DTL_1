import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

image_path = "4.jpg"  # Path to your test image
img = load_img(image_path, target_size=(256, 256))  

img_array = img_to_array(img)  # Convert to array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize if your model expects scaled inputs

model_new = tf.keras.models.load_model('potatoes.h5',compile=False)  

model_new.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 


predictions = model_new.predict(img_array)
class_names = ['Early Blight', 'Late Blight','Healthy']  # Replace with your actual class names

predicted_class = np.argmax(predictions, axis=-1)  # Get the class index
predicted_label = class_names[predicted_class[0]]  # Map index to class name

print(f"Predicted class: {predicted_class}")
print(f"Predicted label: {predicted_label}")