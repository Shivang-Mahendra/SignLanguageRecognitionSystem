import pickle
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

model = load_model(r'D:\MajorProject\New folder\SignLanguageRecognitionSystem\backend\model\modellstm.h5')
with open(r'D:\MajorProject\New folder\SignLanguageRecognitionSystem\backend\model\label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'Q', 17:'R', 18:'S', 19:'T', 20:'U', 21:'V', 22:'W', 23:'X', 24:'Y', 25:'Z'}

while True:

    data_aux = []
    x_ = []
    y_ = []
    z_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
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

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10


        def padding_data(item, length):
            item = list(item)
            if len(item) > length:
                return item[:length]
            else:
                return item + [0] * (length - len(item))


        padded_data = padding_data(data_aux, 63)
        input_data = np.asarray(padded_data).reshape(1, 1, 63).astype(np.float32)

        prediction = model.predict(input_data)

        predicted_class_index = np.argmax(prediction[0])
        predicted_character = label_encoder.inverse_transform([predicted_class_index])[0]
        print(prediction)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, labels_dict[int(predicted_character)], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)


cap.release()
cv2.destroyAllWindows()