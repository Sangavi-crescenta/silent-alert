#!/usr/bin/env python
# coding: utf-8

# **Install Libraries**

# In[20]:


get_ipython().system('pip install tensorflow==2.9.3')


# In[6]:


get_ipython().system('pip install mediapipe==0.10.8')


# In[1]:


get_ipython().system('pip show protobuf')


# In[2]:


get_ipython().system('pip show mediapipe')


# **Extract Videos**

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


get_ipython().system("unrar x '/content/drive/MyDrive/Silent_Alert/Dataset_New.rar'")


# **Collect Video Paths**

# In[3]:


import cv2
import os

# Define paths
base_path = r'/content/Dataset_New'
folders = ['0', '1']

def load_videos():
    video_paths = []
    labels=[]
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        for video_file in os.listdir(folder_path):
            if video_file.endswith('.mp4'):  # Assuming videos are in .mp4 format
                video_paths.append(os.path.join(folder_path, video_file))
                labels.append(int(folder))
    return video_paths,labels

video_paths,labels = load_videos()
print("Total videos found:", len(video_paths))
print("Total Labels found:",len(labels))

print(video_paths[:10],labels[:10])



# In[ ]:





# In[ ]:


from collections import Counter

# Assuming 'labels' is your list of labels (0 or 1)
label_counts = Counter(labels)

label_counts


# **Check Video Landmark Detection**

# In[ ]:


import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Indices for the right hand in Holistic (21 landmarks)
right_hand_landmark_count = 21  # Holistic uses 21 landmarks for each hand

def extract_right_hand_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    right_hand_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB as MediaPipe requires RGB input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        if results.right_hand_landmarks:
            # Extract the right hand landmarks
            frame_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
            right_hand_landmarks.append(frame_landmarks)
        else:
            # If no right hand landmarks detected, add zeros
            right_hand_landmarks.append(np.zeros(right_hand_landmark_count * 3))  # 21 landmarks with x, y, z coordinates

    cap.release()
    return np.array(right_hand_landmarks)  # Shape: (num_frames, 63)

# Example usage:
# landmarks = extract_right_hand_landmarks_from_video("path_to_your_video.mp4")
# print(landmarks.shape)  # Should output: (num_frames, 63)


# In[ ]:


# Example: Extract landmarks from the first video
landmarks_sample = extract_right_hand_landmarks_from_video(video_paths[0])
print("Shape of landmarks:", landmarks_sample.shape)


# **Convert to default landmark count**

# In[ ]:


import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Indices for the right hand in Holistic (21 landmarks)
right_hand_landmark_count = 21  # Holistic uses 21 landmarks for each hand

def extract_right_hand_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    right_hand_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB as MediaPipe requires RGB input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        if results.right_hand_landmarks:
            # Extract the right hand landmarks
            frame_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
            right_hand_landmarks.append(frame_landmarks)
        else:
            # If no right hand landmarks detected, add zeros
            right_hand_landmarks.append(np.zeros(right_hand_landmark_count * 3))  # 21 landmarks with x, y, z coordinates

    cap.release()

    # Convert landmarks list to numpy array
    landmarks = np.array(right_hand_landmarks)

    # Create a zero-filled array of shape [500, 99]
    default_landmarks = np.zeros((500, 63))

    # Place original landmarks in the beginning rows
    num_frames = min(landmarks.shape[0], 500)
    default_landmarks[:num_frames, :] = landmarks[:num_frames]

    return default_landmarks


# In[ ]:


# Example usage
video_path = video_paths[0]
landmarks_sample = extract_right_hand_landmarks_from_video(video_path)
print("Shape of default landmarks:", landmarks_sample.shape)


# In[ ]:


landmarks_sample


# In[ ]:


landmarks_sample[:83]


# **Collect All Video Landmarks**

# In[5]:


import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Indices for the right hand in Holistic (21 landmarks)
right_hand_landmark_count = 21  # Holistic uses 21 landmarks for each hand

def extract_right_hand_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    right_hand_landmarks = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB as MediaPipe requires RGB input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        if results.right_hand_landmarks:
            # Extract the right hand landmarks
            frame_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
            right_hand_landmarks.append(frame_landmarks)
        else:
            # If no right hand landmarks detected, add zeros
            right_hand_landmarks.append(np.zeros(right_hand_landmark_count * 3))  # 21 landmarks with x, y, z coordinates

    cap.release()

    # Convert landmarks list to numpy array
    landmarks = np.array(right_hand_landmarks)

    # Create a zero-filled array of shape [500, 99]
    default_landmarks = np.zeros((500, 63))

    # Place original landmarks in the beginning rows
    num_frames = min(landmarks.shape[0], 500)
    default_landmarks[:num_frames, :] = landmarks[:num_frames]

    return default_landmarks

# Process all videos in a given directory and label them
def process_videos(video_paths, labels):
    data = []
    for i, video_path in enumerate(video_paths):
        print(i)
        landmarks = extract_right_hand_landmarks_from_video(video_path)
        data.append((landmarks, labels[i]))  # Append tuple (landmarks, label)
    return data

# Example usage
video_paths = video_paths  # Add your video paths here


# Extract landmarks and labels for all videos
data = process_videos(video_paths, labels)

# Separate landmarks and labels for training
landmarks_array = np.array([item[0] for item in data])  # Shape: [num_videos, 500, 99]
labels_array = np.array([item[1] for item in data])  # Shape: [num_videos]




# **Save Landmarks and Labels in numpy array**

# In[ ]:


np.save('/content/drive/MyDrive/Silent_Alert/landmarks_600_600.npy', landmarks_array)
np.save('/content/drive/MyDrive/Silent_Alert/labels_600_600.npy', labels_array)

print("Saved landmarks and labels as .npy files.")


# In[9]:


print(landmarks_array.shape)
print(labels_array.shape)


# **Importing Libraries**

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Masking


# **Load Landmark and Label Data**

# In[3]:


landmarks = np.load('landmarks_600_600.npy')  # Shape: [num_videos, 500, 99]
labels = np.load('labels_600_600.npy')  # Shape: [num_videos]


# **Convert labels to categorical**

# In[4]:


labels = tf.keras.utils.to_categorical(labels, num_classes=2)


# In[5]:


labels[500:1000]


# **BiLSTM model**

# In[6]:


model = Sequential([
    Masking(mask_value=0.0, input_shape=(500, 63)),  # Mask padding of zeros
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(64)),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')  # Adjust units if you have more classes
])


# In[7]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# **Training**

# In[ ]:


# Train the model
history=model.fit(landmarks, labels, epochs=100, batch_size=16, validation_split=0.2)


# **Save Model**

# In[ ]:


model.save("/content/drive/MyDrive/Silent_Alert/bilstm_model.h5")


# **Metrics**

# In[8]:


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


# **Confusion Matrixx**

# In[9]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("/content/drive/MyDrive/Silent_Alert/bilstm_model.h5")

# Load the landmarks and labels
landmarks = np.load('/content/drive/MyDrive/Silent_Alert/landmarks_600_600.npy')
labels = np.load('/content/drive/MyDrive/Silent_Alert/labels_600_600.npy')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(landmarks, labels, test_size=0.2, random_state=42)

# Make predictions on the test set
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)  # Get predicted class labels

# Convert y_test to class labels if it's one-hot encoded
if len(y_test.shape) > 1 and y_test.shape[1] > 1:
    y_test = np.argmax(y_test, axis=1)


# Calculate the test accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy}")

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Class 0', 'Class 1'],
            yticklabels=['Class 0', 'Class 1']) # Assuming two classes
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# **Live Prediction**

# In[4]:


import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Initialize MediaPipe Pose and OpenCV
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)


mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Load the trained BiLSTM model

global landmarks_buffer
# Initialize camera and buffer
cap = cv2.VideoCapture(0)
landmarks_buffer = []
lm_cnt=[]

global label

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB as MediaPipe requires RGB input
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)
    label = "waiting"
    if results.right_hand_landmarks is None:
        lm_cnt.append(0)

    if results.right_hand_landmarks:
        # Extract the right hand landmarks
        frame_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
        if frame_landmarks is not None:

            # Start collecting landmarks if a hand is detected
            landmarks_buffer.append(frame_landmarks)
            lm_cnt.append(1)
            label = "Hand detected, collecting landmarks"

    def predict(padded_landmarks):
        # Reshape for the model and make a prediction if we have 500 frames
        input_data = padded_landmarks.reshape(1, 500, 63)
        model = load_model('bilstm_model.h5')
        prediction = model.predict(input_data) # Assuming binary classification with a single output node

        # Get predicted label
        #predicted_label = np.argmax(prediction) if prediction.shape[-1] > 1 else int(prediction[0] > 0.5)

        print(prediction)
        if prediction[0][np.argmax(prediction[0])] > 0.95:
            predicted_label = np.argmax(prediction)
            # Set label based on prediction
            if predicted_label == 1:
                label = 'Emergency'
            else:
                label = 'Rescue'
        else:
              label="Not Accurate"
        landmarks_buffer.clear()
        lm_cnt.clear()


        # Clear landmarks buffer after prediction to start fresh on next hand detection

        return prediction,label

    if landmarks_buffer:
        landmarks=np.array(landmarks_buffer).shape
        # Pad the sequence to 500 frames if it's shorter, and check for prediction


    if len(lm_cnt)> 10 and sum(lm_cnt)>10:
        if lm_cnt[-4:]==[1,0,0,0]:
            if len(landmarks_buffer) < 500:
                padded_landmarks = np.vstack([landmarks_buffer, np.zeros((500 - len(landmarks_buffer), 63))])
                print(padded_landmarks.shape)
            else:
                padded_landmarks = np.array(landmarks_buffer[-500:])
            if padded_landmarks.shape[0] == 500:
                predicted_Valus,label=predict(padded_landmarks)

    #print(predicted_Valus)
    # Display the camera feed and prediction result
    if label != "waiting" and label != "Hand detected, collecting landmarks":
        print("Prediction Class:",label)
    cv2.putText(frame, f"Status: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Live Prediction', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[5]:


print(lm_cnt)


# In[6]:


sum(lm_cnt)


# In[7]:


padded_landmarks.shape

