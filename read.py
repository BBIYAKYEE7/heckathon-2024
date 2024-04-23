import cv2
import numpy as np
import subprocess
import mediapipe as mp

def recognize_hand_motion(frame):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    results = hands.process(frame)
    
    if results.multi_hand_landmarks:
        angles = []
        for hand_landmarks in results.multi_hand_landmarks:
            hand_angles = []
            for landmark in hand_landmarks.landmark:
                hand_angles.append((landmark.x, landmark.y, landmark.z))
            angles.append(hand_angles)
        return angles
    return None

# 파일에서 데이터셋 불러오기
file_path = 'hand_motion_angles.txt'
with open(file_path, 'r') as file:
    dataset_angles = [list(map(float, line.strip().split(','))) for line in file]

# 임계값 설정
threshold = 5

# 웹캠 열기
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # 손 모션 인식
    current_angles = recognize_hand_motion(frame)
    
    if current_angles is not None:
        # 현재 손 모션의 각도와 데이터셋의 각도와 비교하여 일치하는 경우 크롬을 실행
        for hand_angles in current_angles:
            # 각 손가락의 각도를 개별적으로 비교하고 일치하는지 확인합니다.
            if np.allclose(hand_angles, dataset_angles, atol=threshold):
                # 크롬 실행 명령
                subprocess.Popen(["C:/Program Files/Google/Chrome/Application/chrome.exe"])
                break

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
