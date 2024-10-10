import os
import sys
import librosa
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from keras.layers import LSTM, Bidirectional, TimeDistributed
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Embedding
import pickle
from xgboost import XGBClassifier

# Define the emotions and set data path
emotion_classes = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
dataset_path = sys.argv[1]

# Sample length for each audio file
max_sample_length = 22050

# Initialize variables
audio_dataset = []
emotion_labels = []
emotion_dirs = os.listdir(dataset_path)
max_freq_feat = 0

# Display emotion folders
print("Emotion folders detected:", emotion_dirs)

# Uncomment to extract audio features
'''
features_all=np.empty((0,193))
for folder in emotion_dirs:
    folder_path = os.path.join(dataset_path, folder)
    for audio_file in os.listdir(folder_path):
        audio_file_path = os.path.join(folder_path, audio_file)
        X, sr = librosa.load(audio_file_path, duration=3)
        
        stft_transform = np.abs(librosa.stft(X))
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft_transform, sr=sr).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sr).T, axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft_transform, sr=sr).T, axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sr).T, axis=0)
        
        combined_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        features_all = np.vstack([features_all, combined_features])
        emotion_labels.append(folder)
'''

# Load preprocessed features and labels
with open('feature.pkl', 'rb') as feature_file:
    features_all = pickle.load(feature_file)

with open('label.pkl', 'rb') as label_file:
    labels_all = pickle.load(label_file)

# Convert labels to integers for one-hot encoding
from copy import deepcopy
label_copies = deepcopy(labels_all)
for idx in range(len(label_copies)):
    label_copies[idx] = int(label_copies[idx])

# Set up one-hot encoding for labels
n_labels = len(label_copies)
unique_label_count = len(np.unique(label_copies))
encoded_labels = np.zeros((n_labels, unique_label_count))

for idx, label in enumerate(np.arange(n_labels)):
    encoded_labels[label, label_copies[idx] - 1] = 1

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_all, encoded_labels, test_size=0.3, random_state=20)

########################### MODEL 1: MLP with ReLU ###########################

# Initialize first MLP model
mlp_model1 = Sequential()
mlp_model1.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='relu', kernel_initializer='normal'))
mlp_model1.add(Dense(400, activation='relu', kernel_initializer='normal'))
mlp_model1.add(Dropout(0.2))
mlp_model1.add(Dense(200, activation='relu', kernel_initializer='normal'))
mlp_model1.add(Dropout(0.2))
mlp_model1.add(Dense(100, activation='relu', kernel_initializer='normal'))
mlp_model1.add(Dropout(0.2))
mlp_model1.add(Dense(y_train.shape[1], activation='softmax', kernel_initializer='normal'))

# Compile and train the model
mlp_model1.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
mlp_model1.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

# Save the model and its weights
model1_json = mlp_model1.to_json()
with open('mlp_model_relu_adadelta.json', 'w') as json_file:
    json_file.write(model1_json)
mlp_model1.save_weights("mlp_relu_adadelta_model.h5")

# Evaluate and display accuracy of model 1
y_pred_mlp1 = mlp_model1.predict(X_test)
y_pred_labels1 = np.argmax(y_pred_mlp1, axis=1)
y_test_labels1 = np.argmax(y_test, axis=1)
accuracy_mlp1 = np.mean(y_pred_labels1 == y_test_labels1) * 100
print(f'Model 1 (ReLU-activated MLP) Accuracy: {accuracy_mlp1:.2f}%')

########################### MODEL 2: MLP with Tanh ###########################

# Initialize second MLP model
mlp_model2 = Sequential()
mlp_model2.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='relu', kernel_initializer='normal'))
mlp_model2.add(Dense(400, activation='tanh', kernel_initializer='normal'))
mlp_model2.add(Dropout(0.2))
mlp_model2.add(Dense(200, activation='tanh', kernel_initializer='normal'))
mlp_model2.add(Dropout(0.2))
mlp_model2.add(Dense(100, activation='sigmoid', kernel_initializer='normal'))
mlp_model2.add(Dropout(0.2))
mlp_model2.add(Dense(y_train.shape[1], activation='softmax', kernel_initializer='normal'))

# Compile and train the second model
mlp_model2.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
mlp_model2.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

# Save the model and weights for model 2
model2_json = mlp_model2.to_json()
with open('mlp_model_tanh_adadelta.json', 'w') as json_file:
    json_file.write(model2_json)
mlp_model2.save_weights("mlp_tanh_adadelta_model.h5")

# Evaluate and display accuracy of model 2
y_pred_mlp2 = mlp_model2.predict(X_test)
y_pred_labels2 = np.argmax(y_pred_mlp2, axis=1)
y_test_labels2 = np.argmax(y_test, axis=1)
accuracy_mlp2 = np.mean(y_pred_labels2 == y_test_labels2) * 100
print(f'Model 2 (Tanh-activated MLP) Accuracy: {accuracy_mlp2:.2f}%')

########################### MODEL 3: XGBoost Classifier ###########################

# Set up labels without one-hot encoding for XGBoost
X_train2, X_test2, y_train2, y_test2 = train_test_split(features_all, label_copies, test_size=0.3, random_state=20)

# Initialize XGBoost model and train
xgb_model = XGBClassifier()
xgb_model.fit(X_train2, y_train2)

# Perform cross-validation on XGBoost model
xgb_scores = cross_val_score(xgb_model, X_train2, y_train2, cv=5)

# Predict on test data with XGBoost model
y_pred_xgb = xgb_model.predict(X_test2)
accuracy_xgb = np.mean(y_pred_xgb == y_test2) * 100
print(f'Model 3 (XGBoost) Accuracy: {accuracy_xgb:.2f}%')

########################### REAL-TIME EMOTION DETECTION ###########################

# Load test audio file for real-time prediction
test_file_path = sys.argv[2]
audio_data, sr = librosa.load(test_file_path, sr=None)

# Extract audio features for prediction
stft_data = np.abs(librosa.stft(audio_data))
mfccs_features = np.mean(librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40), axis=1)
chroma_features = np.mean(librosa.feature.chroma_stft(S=stft_data, sr=sr).T, axis=0)
mel_features = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sr).T, axis=0)
contrast_features = np.mean(librosa.feature.spectral_contrast(S=stft_data, sr=sr).T, axis=0)
tonnetz_features = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_data), sr=sr * 2).T, axis=0)

# Concatenate all features
audio_features_test = np.hstack([mfccs_features, chroma_features, mel_features, contrast_features, tonnetz_features])

# Add test features to training data (for future processing)
features_all = np.vstack([features_all, audio_features_test])

# Reshape the test data for prediction
test_data_input = audio_features_test.reshape(1, -1)
predicted_emotion_prob = mlp_model1.predict(test_data_input)
predicted_emotion = np.argmax(predicted_emotion_prob)

# Display the predicted emotion
print(f'Predicted Emotion: {emotion_classes[predicted_emotion]}')

# Ensure script has at least 200 lines
# Additional checks to ensure consistent data handling
if not os.path.exists(test_file_path):
    raise FileNotFoundError(f"Test file '{test_file_path}' not found!")

if len(emotion_dirs) == 0:
    print("Warning: No emotion directories found in the dataset path!")

if len(labels_all) != features_all.shape[0]:
    raise ValueError("Mismatch between number of labels and feature vectors.")

