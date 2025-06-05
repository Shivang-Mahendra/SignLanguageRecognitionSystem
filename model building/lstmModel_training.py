import pickle
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# dataset splitting
data_dict = pickle.load(open('./data.pickle', 'rb'))

def padding_data(item, length):
    item = list(item)
    if len(item) > length:
        return item[:length]
    else:
        return item + [0] * (length - len(item))

padded_data = [padding_data(item, 63) for item in data_dict['data']]

data = np.asarray(padded_data)
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)

x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)

x_train = np.array(x_train, dtype=np.float32)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train)
y_categorical = to_categorical(y_encoded)
y_categorical = y_categorical.astype(np.float32)
y_test_encoded = label_encoder.transform(y_test)
y_test_encoded = y_test_encoded.astype(np.float32)

x_train = np.nan_to_num(x_train).astype(np.float32)
y_categorical = np.nan_to_num(y_categorical).astype(np.float32)

# debugging
print("x_train shape:", x_train.shape)
print("x_train dtype:", x_train.dtype)
print("y_categorical shape:", y_categorical.shape)
print("y_categorical dtype:", y_categorical.dtype)
print("y_test dtype:", y_test.dtype)
print("y_test_encoded dtype:", y_test_encoded.dtype)
print("Any NaNs in x_train?", np.isnan(x_train).any())
print("Any NaNs in y_categorical?", np.isnan(y_categorical).any())


# LSTM model building
num_classes = y_categorical.shape[1]
sequence_length = 1
feature_dimensions = 63

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(sequence_length, feature_dimensions)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# model training
history = model.fit(x_train, y_categorical, epochs=300, batch_size=32, validation_split=0.2)

#Saving training history
with open('model_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)

with open('model_history.pkl', 'rb') as f:
    history_data = pickle.load(f)


prediction = model.predict(x_test)
predicted_classes = np.argmax(prediction, axis=1)

score = accuracy_score(predicted_classes, y_test_encoded)
print('{}% of samples were classified correctly !'.format(score * 100))

y_true = np.argmax(to_categorical(LabelEncoder().fit_transform(y_test)), axis=1)
cm = confusion_matrix(y_true, predicted_classes)
print("Classification Report:")
print(classification_report(y_true, predicted_classes))

#Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[chr(i) for i in range(65, 91)],
            yticklabels=[chr(i) for i in range(65, 91)])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)  # You can use 'pdf' instead
plt.close()


plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=300)  # High resolution
plt.close()

# saving model for further use
model.save('modellstm.h5')

#saving label encoder to recognize hand signs
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)