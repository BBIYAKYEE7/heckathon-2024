import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# 파일에 손 모션의 각도를 저장하기 위한 변수
file_path = 'hand_motion_angles.txt'
file = open(file_path, 'w')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 각 관절의 3차원 좌표를 저장할 빈 리스트
            joint_coordinates = []
            for landmark in hand_landmarks.landmark:
                # 손의 각 관절의 3차원 좌표를 리스트에 추가
                joint_coordinates.append((landmark.x, landmark.y, landmark.z))

            # 손가락 각도 계산
            angles = []
            for i in range(1, 5):  # 손가락은 총 5개
                for j in range(i+1, 5):  # 손가락 끝점까지의 각도를 계산
                    # 각 손가락 끝점과 중심에서의 벡터 계산
                    vector1 = np.array(joint_coordinates[i*4]) - np.array(joint_coordinates[i*4-3])
                    vector2 = np.array(joint_coordinates[j*4]) - np.array(joint_coordinates[i*4-3])
                    # 내적을 이용하여 각도 계산
                    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
                    angle = np.arccos(cosine_angle)
                    angle = np.degrees(angle)
                    angles.append(angle)

            # 파일에 손 모션의 각도를 추가
            file.write(','.join(map(str, angles)) + '\n')

            # 시각화를 위해 손 모양 그리기
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 파일을 닫음
file.close()

hands.close()
cap.release()
cv2.destroyAllWindows()
