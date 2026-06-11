from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io

# Import TensorFlow/Keras
from tensorflow.keras.models import load_model
# IMPORT THE CORRECT PREPROCESSING FUNCTION
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
MODEL_PATH = 'best_model.keras'

# CORRECT ALPHABETICAL ORDER (image_dataset_from_directory sorts this way)
# Heart (H) -> Oblong (O..b) -> Oval (O..v) -> Round (R) -> Square (S)
CLASS_LABELS = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

IMG_SIZE = (224, 224)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
print("Loading model...")
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        # 1. Read the image
        img = Image.open(io.BytesIO(file.read()))
        
        # 2. Preprocess
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img = img.resize(IMG_SIZE)
        img_array = np.array(img)
        
        # CRITICAL FIX: Use EfficientNet preprocess_input instead of / 255.0
        img_array = preprocess_input(img_array)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # 3. Predict
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        probs = predictions[0].tolist()

        # 4. Format Result
        result = {
            'face_shape': CLASS_LABELS[predicted_index],
            'confidence': round(confidence * 100, 2),
            'probabilities': {
                'heart': round(probs[0] * 100, 2),
                'oblong': round(probs[1] * 100, 2),
                'oval': round(probs[2] * 100, 2),
                'round': round(probs[3] * 100, 2),
                'square': round(probs[4] * 100, 2)
            }
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)