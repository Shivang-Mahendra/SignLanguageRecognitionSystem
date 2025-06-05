from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
import mediapipe as mp
import pickle
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Load model and label encoder
model = load_model(r'D:\MajorProject\New folder\SignLanguageRecognitionSystem\backend\model\modellstm.h5')
with open(r'D:\MajorProject\New folder\SignLanguageRecognitionSystem\backend\model\label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# to get charcters from predicted labels
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M',
               13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Sentence storage
sentence = []

def extract_landmarks(image):

    # collect hand landmarks
    data_aux = []
    x_ = []
    y_ = []
    z_ = []
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z

                x_.append(x)
                y_.append(y)
                z_.append(z)
            
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                z = hand_landmarks.landmark[i].z

                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))
                data_aux.append(z - min(z_))
        
        return data_aux
    
    return None


def padding_data(item, length):
    item = list(item)
    if len(item) > length:
        return item[:length]
    else:
        return item + [0] * (length - len(item))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image'].split(',')[1]  # remove base64 header
    decoded = base64.b64decode(image_data)
    nparr = np.frombuffer(decoded, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Extract landmarks from image
    landmarks = extract_landmarks(img)
    if landmarks is None:
        return jsonify({'character': ''})  # no hand detected

    # Prepare input for model
    padded_data = padding_data(landmarks, 63)
    input_data = np.asarray(padded_data).reshape(1, 1, 63).astype(np.float32)

    # Predict
    y_pred = model.predict(input_data)
    predicted_class = np.argmax(y_pred, axis=1)[0]

    # Convert class index to character label
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    
    return jsonify({'character': labels_dict[int(predicted_label)]})


@app.route('/clear', methods=['POST'])
def clear():
    global sentence
    sentence = []
    return jsonify({'status': 'cleared'})


if __name__ == '__main__':
    app.run(debug=True)
