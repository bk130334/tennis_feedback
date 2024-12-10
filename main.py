import os
import cv2
import numpy as np
import mediapipe as mp
import joblib
import tkinter as tk
from tkinter import messagebox

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Step 1: Loading model and scaler...")
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    exit()

print("Step 2: Initializing Mediapipe...")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3)

def get_feedback(landmarks):
    feedback = []
    if landmarks[0] < 0.3:  # 왼쪽 어깨가 너무 낮은 경우
        feedback.append("왼쪽 어깨를 올리십시오.")
    elif landmarks[0] > 0.7:  # 왼쪽 어깨가 너무 높은 경우
        feedback.append("왼쪽 어깨를 내리십시오.")
    else:
        feedback.append("왼쪽 어깨 위치가 적당합니다.")

    if landmarks[1] < 0.3:  # 오른쪽 어깨가 너무 낮은 경우
        feedback.append("오른쪽 어깨를 올리십시오.")
    elif landmarks[1] > 0.7:  # 오른쪽 어깨가 너무 높은 경우
        feedback.append("오른쪽 어깨를 내리십시오.")
    else:
        feedback.append("오른쪽 어깨 위치가 적당합니다.")

    if landmarks[2] < 0.3:  # 왼쪽 손목이 너무 낮은 경우
        feedback.append("왼쪽 손목을 올리십시오.")
    elif landmarks[2] > 0.7:  # 왼쪽 손목이 너무 높은 경우
        feedback.append("왼쪽 손목을 내리십시오.")
    else:
        feedback.append("왼쪽 손목 위치가 적당합니다.")

    if landmarks[3] < 0.3:  # 오른쪽 손목이 너무 낮은 경우
        feedback.append("오른쪽 손목을 올리십시오.")
    elif landmarks[3] > 0.7:  # 오른쪽 손목이 너무 높은 경우
        feedback.append("오른쪽 손목을 내리십시오.")
    else:
        feedback.append("오른쪽 손목 위치가 적당합니다.")

    return "\n".join(feedback)

def preprocess_image(image_path):
    print("Step 2.1: Checking image path...")
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    print("Step 2.2: Loading image...")
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to read the image.")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("Step 2.3: Detecting pose...")
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print("No pose detected in the image.")
        return None

    keypoints = []
    keypoints_indices = [15, 16, 12, 11, 13, 14, 0] 
    for index in keypoints_indices:
        lm = results.pose_landmarks.landmark[index]
        keypoints.append(lm.x)
        keypoints.append(lm.y)
        keypoints.append(lm.z)

    print("Pose detected successfully.")
    print(f"Keypoints shape: {np.array(keypoints).shape}")
    return np.array(keypoints).reshape(1, -1), keypoints 

def adjust_probabilities(probabilities):
    """
    확률값을 보정하는 함수.
    serve와 backhand의 확률을 낮추고 다른 클래스의 확률을 상대적으로 높임.
    """
    adjusted_probabilities = probabilities.copy()

    # serve와 backhand 확률 보정 (serve * 0.4, backhand * 0.5)
    adjusted_probabilities['serve'] = adjusted_probabilities['serve'] * 0.4
    adjusted_probabilities['backhand'] = adjusted_probabilities['backhand'] * 0.5

    total_adjusted = sum(adjusted_probabilities.values())
    if total_adjusted > 0:
        for key in adjusted_probabilities:
            adjusted_probabilities[key] /= total_adjusted

    return adjusted_probabilities

def predict_pose(image_path):
    print("Step 3: Processing image for prediction...")
    features, landmarks = preprocess_image(image_path)
    if features is None:
        return "Prediction failed: No pose detected.", None, None

    print(f"Extracted features: {features}")
    print(f"Landmarks: {landmarks}")
    
    print("Step 4: Scaling features...")
    try:
        scaled_features = scaler.transform(features)
        print(f"Scaled features: {scaled_features}")
    except Exception as e:
        print(f"Error scaling features: {e}")
        return "Prediction failed during scaling.", None, None

    print("Step 5: Predicting pose...")
    try:
        prediction_probabilities = model.predict_proba(scaled_features)
        prediction = model.predict(scaled_features)
        print(f"Prediction result: {prediction}")
        print(f"Prediction probabilities: {prediction_probabilities}")

        labels = {0: 'serve', 1: 'stand', 2: 'forehand', 3: 'backhand'}
        
        probabilities = {labels[i]: prediction_probabilities[0][i] for i in range(4)}

        adjusted_probabilities = adjust_probabilities(probabilities)

        predicted_class = max(adjusted_probabilities, key=adjusted_probabilities.get)

        return predicted_class, adjusted_probabilities, landmarks
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Prediction failed.", None, None

def show_result_ui(prediction, probabilities, feedback):
    root = tk.Tk()
    root.title("Pose Prediction")

    label = tk.Label(root, text=f"입력된 자세는 {prediction}입니다.", font=("Helvetica", 16))
    label.pack(pady=20)

    prob_text = "\n".join([f"{key}: {value*100:.2f}%" for key, value in probabilities.items()])
    prob_label = tk.Label(root, text=f"보정된 확률:\n{prob_text}", font=("Helvetica", 12))
    prob_label.pack(pady=10)

    feedback_label = tk.Label(root, text=f"피드백:\n{feedback}", font=("Helvetica", 12))
    feedback_label.pack(pady=10)

    def close_window():
        root.quit() 

    button = tk.Button(root, text="확인", command=close_window, font=("Helvetica", 12))
    button.pack(pady=10)

    root.mainloop()

def main():
    image_path = "image.jpg"  # 이미지 경로
    print(f"Testing with image: {image_path}")
    result, probabilities, landmarks = predict_pose(image_path)
    print(f"Prediction result: {result}")

    if landmarks is not None:
        feedback = get_feedback(landmarks)
    else:
        feedback = "피드백을 제공할 수 없습니다."

    # UI 출력
    show_result_ui(result, probabilities, feedback)

if __name__ == "__main__":
    main()
