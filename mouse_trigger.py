import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time
import speech_recognition as sr
import pyttsx3
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" 

# 미디어파이프의 Hands 모델 로드
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_face_mesh = mp.solutions.face_mesh  # 얼굴 인식

# 미디어파이프의 Drawing 모듈 로드
mp_drawing = mp.solutions.drawing_utils

# 스크롤 속도 및 임계값 설정
scroll_speed = 30
scroll_threshold = 0
# 초기 설정
baseline_length = None
baseline_error = 23
baseline_set = False

initial_baseline_length = None
initial_baseline_set = False

move_threshold = 10
previous_hand_location = None

# 펜 모드 설정
pen_mode = False
previous_point = None
previous_pen_deactivation_time = None
last_pen_deactivation_time = time.time()


canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# 펜 모드가 시작된 시간과 지난 시간을 추적하는 변수 추가
pen_mode_start_time = None
pen_mode_end_time = None


def find_nose_center(face_landmarks):
    x_sum = 0
    y_sum = 0
    for landmark in face_landmarks.landmark:
        x_sum += landmark.x
        y_sum += landmark.y
    num_landmarks = len(face_landmarks.landmark)
    nose_center_x = int(x_sum / num_landmarks * img.shape[1])
    nose_center_y = int(y_sum / num_landmarks * img.shape[0])
    return nose_center_x, nose_center_y

def find_face_center(face_landmarks):
    x_sum = 0
    y_sum = 0
    for landmark in face_landmarks.landmark:
        x_sum += landmark.x
        y_sum += landmark.y
    num_landmarks = len(face_landmarks.landmark)
    face_center_x = int(x_sum / num_landmarks * img.shape[1])
    face_center_y = int(y_sum / num_landmarks * img.shape[0])
    return face_center_x, face_center_y

def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def draw_line_with_length(img, point1, point2, line_color, text_color, thickness):
    # 선을 그림
    cv2.line(img, point1, point2, line_color, thickness)
    
    # 두 점 사이의 거리를 계산
    length = int(math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2))
    
    # 텍스트를 선의 중간에 표시
    mid_point = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
    cv2.putText(img, str(length), mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    
   
# 음성 인식기 생성
recognizer = sr.Recognizer()
korean_voice_path = "korean_voice.mp3"

# 음성 출력 엔진 생성
engine = pyttsx3.init()

# 음성을 텍스트로 변환하는 함수
def recognize_speech():
    with sr.Microphone() as source:
        print("말하세요!")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language='ko-KR')
        print("인식된 내용:", text)
        return text
    except sr.UnknownValueError:
        print("음성을 이해하지 못했습니다.")
        return None
    except sr.RequestError as e:
        print("결과를 가져올 수 없습니다; {0}".format(e))
        return None

# 텍스트를 음성으로 출력하는 함수
def speak_text(text):
    engine.say(text)
    engine.runAndWait()
    
# 비디오 캡처
cap = cv2.VideoCapture(0)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    # BGR에서 RGB로 변환
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 미디어파이프를 사용하여 손 검출
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        left_hand_landmarks = None
        right_hand_landmarks = None

        # 각 손의 관절을 추출
        for hand_landmarks in results.multi_hand_landmarks:
            if hand_landmarks.landmark[0].x < hand_landmarks.landmark[17].x:  # 첫 번째 손 (왼손)
                left_hand_landmarks = hand_landmarks
            else:  # 두 번째 손 (오른손)
                right_hand_landmarks = hand_landmarks

        # 왼손 관절 표시
        if left_hand_landmarks:
            # 왼손 위치 정보에서 최솟값과 최댓값을 찾아 박스의 좌표 계산
            x_min = min(int(landmark.x * img.shape[1]) for landmark in left_hand_landmarks.landmark)
            x_max = max(int(landmark.x * img.shape[1]) for landmark in left_hand_landmarks.landmark)
            y_min = min(int(landmark.y * img.shape[0]) for landmark in left_hand_landmarks.landmark)
            y_max = max(int(landmark.y * img.shape[0]) for landmark in left_hand_landmarks.landmark)

            # 계산된 좌표를 사용하여 박스 그리기
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # 빨간색 박스로 변경

            for idx, landmark in enumerate(left_hand_landmarks.landmark):
                # 각 관절의 좌표
                cx, cy = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                cv2.putText(img, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # 왼손의 검지와 엄지 연결
            index_tip_left = (int(left_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * img.shape[1]),
                            int(left_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * img.shape[0]))
            thumb_tip_left = (int(left_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * img.shape[1]),
                            int(left_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * img.shape[0]))
            middle_tip_left = (int(left_hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * img.shape[1]),
                            int(left_hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * img.shape[0]))
            
            draw_line_with_length(img, index_tip_left, thumb_tip_left, (255, 255, 255), (0, 0, 255), 2)
            draw_line_with_length(img, middle_tip_left, thumb_tip_left, (255, 255, 255), (0, 0, 255), 2)

        # 오른손 관절 표시
        if right_hand_landmarks:
            # 오른손 위치 정보에서 최솟값과 최댓값을 찾아 박스의 좌표 계산
            x_min = min(int(landmark.x * img.shape[1]) for landmark in right_hand_landmarks.landmark)
            x_max = max(int(landmark.x * img.shape[1]) for landmark in right_hand_landmarks.landmark)
            y_min = min(int(landmark.y * img.shape[0]) for landmark in right_hand_landmarks.landmark)
            y_max = max(int(landmark.y * img.shape[0]) for landmark in right_hand_landmarks.landmark)

            # 계산된 좌표를 사용하여 박스 그리기
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # 빨간색 박스로 변경
            
            # 오른손 랜드마크 인덱스를 21부터 시작하여 표시
            for idx, landmark in enumerate(right_hand_landmarks.landmark):
                # 각 관절의 좌표
                cx, cy = int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])
                cv2.putText(img, str(idx + 21), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
       

            # 오른손의 검지와 엄지 끝 좌표 계산
            index_tip_right = (int(right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * img.shape[1]),
                            int(right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * img.shape[0]))
            thumb_tip_right = (int(right_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * img.shape[1]),
                            int(right_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * img.shape[0]))
            middle_tip_right = (int(right_hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * img.shape[1]),
                                int(right_hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * img.shape[0]))
            


            distance_index_thumb = calculate_distance(middle_tip_right, thumb_tip_right)
            
            start_time = time.time()  # 음성 인식 시작 시간 기록

            while distance_index_thumb <= 30:
                if time.time() - start_time >= 3:
                    # 음성 인식이 3초 이상 지속되지 않으면 취소
                    break

                # 음성 인식 시작
                recognized_text = recognize_speech()
                
                if recognized_text:
                    # 텍스트 출력
                    cv2.putText(img, "인식된 내용: " + recognized_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            distance_index_thumb = calculate_distance(index_tip_right, thumb_tip_right)
            

            # 손가락 끝 좌표 및 펜 모드 결정
            if distance_index_thumb <= 35:
                pen_mode = True
            else:
                pen_mode = False

            if pen_mode:
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = img.shape
                cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                


                # 이전 손 위치가 있으면 현재 손 위치와 이전 손 위치를 선으로 연결하여 그립니다.
                if previous_hand_location is not None and previous_hand_location != (cx, cy):
                    # 캔버스에 그려지는 펜의 색상
                    pen_color = (0, 0, 255)  # 진한 파란색

                    # 캔버스에 그려지는 선의 색상
                    cv2.line(canvas, previous_hand_location, (cx, cy), pen_color, 40)  # 이전 손 위치와 현재 손 위치를 선으로 연결하여 그립니다.
                    
                    

                previous_hand_location = (cx, cy)  # 현재 손 위치를 이전 손 위치로 저장합니다.
            else:
                previous_hand_location = None  # 펜을 다시 시작할 때 이전 손 위치를 초기화합니다.
                
                
            draw_line_with_length(img, index_tip_right, thumb_tip_right, (255, 255, 255), (0, 0, 255), 2)  # 흰색 선으로 검지와 엄지 연결
            draw_line_with_length(img, middle_tip_right, thumb_tip_right, (255, 255, 255), (0, 0, 255), 2)  # 흰색 선으로 중간 손가락과 엄지 연결

            # 여기에 추가합니다.
            # 펜 모드가 끊긴 후 3초 이상이 경과하면 그림을 다른 화면에 표시합니다.
            if not pen_mode:
                if pen_mode_end_time is None:
                    pen_mode_end_time = time.time()  # 펜 모드가 끝난 시간 기록
                else:
                    # 현재 시간과 펜 모드가 끝난 시간을 비교하여 3초 이상이 지났는지 확인
                    if time.time() - pen_mode_end_time >= 5:
                        # 펜 모드가 끝난 후 3초 이상이 지났으므로 그림을 다른 화면에 표시합니다.
                        # 새로운 화면 생성 (검정색 배경에 흰색으로 그림을 그릴 예정)
                        display_canvas = np.zeros((480, 640, 3), dtype=np.uint8) # 검정색 배경
                       
                        

                        # 그린 그림 표시 (검정색 배경에 흰색으로)
                        display_canvas = cv2.add(display_canvas, canvas)
                        display_canvas = cv2.flip(display_canvas, 1)
                        
                        # 새 창에 그림 표시
                        cv2.imshow('Drawing', display_canvas)
                        
                        # 펜 모드가 끝났으므로 캔버스 초기화
                        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                        
                        # 펜 모드가 다시 시작되기 전까지 펜 모드 끝 시간 초기화
                        pen_mode_end_time = None
                        
                        

        # 관절을 그리기
        mp_drawing.draw_landmarks(
            img, left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
        )

        mp_drawing.draw_landmarks(
            img, right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
        )

    # 얼굴 메시 처리
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    face_center = None  # 얼굴 중심 초기화

    nose_center = None
    prev_nose_center = None

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 얼굴 주위에 사각형 그리기
            x_min = min(int(coord.x * img.shape[1]) for coord in face_landmarks.landmark)
            x_max = max(int(coord.x * img.shape[1]) for coord in face_landmarks.landmark)
            y_min = min(int(coord.y * img.shape[0]) for coord in face_landmarks.landmark)
            y_max = max(int(coord.y * img.shape[0]) for coord in face_landmarks.landmark)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # 빨간색 박스로 변경

            face_center = find_face_center(face_landmarks)  # 얼굴 중심 저장
            nose_center = find_nose_center(face_landmarks)

    if results.multi_face_landmarks:
        left_face_center = None
        right_face_center = None

        cheek_color = (255, 255, 255)  #흰색

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:
                face_center = find_face_center(face_landmarks)
                if face_center[0] < img.shape[1] // 2:
                    left_face_center = face_center
                else:
                    right_face_center = face_center

            for face_landmarks in results.multi_face_landmarks:
                x_min = min(int(coord.x * img.shape[1]) for coord in face_landmarks.landmark)
                x_max = max(int(coord.x * img.shape[1]) for coord in face_landmarks.landmark)
                y_min = min(int(coord.y * img.shape[0]) for coord in face_landmarks.landmark)
                y_max = max(int(coord.y * img.shape[0]) for coord in face_landmarks.landmark)

                nose_x, _ = find_nose_center(face_landmarks)

                # 얼굴의 왼쪽 볼을 원형으로 표시
                left_cheek_center = ((x_min + nose_x) // 2, (y_min + y_max) // 2)
                left_cheek_radius = min((nose_x - x_min) // 2, (y_max - y_min) // 2)
                cv2.circle(img, (int(left_cheek_center[0]), int(left_cheek_center[1])), int(left_cheek_radius),
                           cheek_color, 2)

                # 얼굴의 오른쪽 볼을 원형으로 표시
                right_cheek_center = ((nose_x + x_max) // 2, (y_min + y_max) // 2)
                right_cheek_radius = min((x_max - nose_x) // 2, (y_max - y_min) // 2)
                cv2.circle(img, (int(right_cheek_center[0]), int(right_cheek_center[1])), int(right_cheek_radius),
                           cheek_color, 2)

                # 원 중심에서 수평선 그리기
                cv2.line(img, (int(left_cheek_center[0] - left_cheek_radius), int(left_cheek_center[1])),
                         (int(right_cheek_center[0] + right_cheek_radius), int(right_cheek_center[1])),
                         (0, 255, 0), 2)
                                    
                    

                # 얼굴 중심과 초록색 수평선 사이의 거리 계산 및 스크롤 조작
                if face_center and left_cheek_center:
                    distance = left_cheek_center[1] - face_center[1]

                    # 일정 거리 이상 움직였을 때만 스크롤 작동
                    if abs(distance) > scroll_threshold and distance != 0:  # 거리가 0이 아닐 때만 스크롤 작동
                        if -4 < distance < 0:
                            # 거의 움직이지 않은 경우 스크롤 작동하지 않음
                            pass
                        elif distance > 0:
                            scroll_speed += 7
                            pyautogui.scroll(scroll_speed)  # 페이지를 올립니다.
                        else:
                            pyautogui.scroll(-scroll_speed)  # 페이지를 내립니다.


    img = cv2.add(img, canvas)  # 두 이미지를 단순히 더합니다.
    img = cv2.flip(img, 1)  # 이미지 좌우 반전
    cv2.imshow('Hand Tracking', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()