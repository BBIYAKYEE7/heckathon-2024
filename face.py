import cv2
import mediapipe as mp
import math
import pyautogui

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)

scroll_speed = 5
scroll_threshold = 0.05  

baseline_length = None
baseline_error = 23
baseline_set = False

prev_face_center = None
prev_face_angle = 0

# 초기에 코의 수평선을 그리기 위한 변수
initial_nose_y = None
initial_baseline_length = None
initial_baseline_set = False

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                nose_landmark = face_landmarks.landmark[5]
                nose_x = int(nose_landmark.x * frame.shape[1])
                nose_y = int(nose_landmark.y * frame.shape[0])
                
                # 초기 코의 위치 저장
                if not initial_baseline_set:
                    initial_nose_y = nose_y
                    initial_baseline_length = nose_y
                    initial_baseline_set = True
                
                nose_error = abs(nose_y - initial_baseline_length)
                
                # 코의 위치가 일정 범위를 벗어나면 초기 위치를 다시 설정
                if nose_error > baseline_error:
                    initial_baseline_length = nose_y
                
                # 초기 코의 위치를 기준으로 수평선 그리기
                cv2.line(frame, (0, initial_baseline_length), (frame.shape[1], initial_baseline_length), (255, 0, 0), 2)

                x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')
                for landmark in face_landmarks.landmark:
                    x, y = landmark.x * frame.shape[1], landmark.y * frame.shape[0]
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                face_center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
                
                face_angle = math.degrees(math.atan2(y_max - y_min, x_max - x_min))
                
                if prev_face_center is not None:
                    nose_movement = nose_y - initial_baseline_length
                    
                    # 스크롤 제어
                    if nose_movement > 0:
                        pyautogui.scroll(-scroll_speed)  
                    elif nose_movement < 0:
                        pyautogui.scroll(scroll_speed) 
                
                prev_face_center = face_center
                prev_face_angle = face_angle
                
                # 얼굴 전체를 인식하는 초록색 박스 그리기
                green_color = (0, 255, 0)  # 초록색
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), green_color, 2)

                # 얼굴의 중심 좌표인 코의 x 좌표를 기준으로 얼굴을 왼쪽과 오른쪽으로 나누기
                nose_x = int(face_landmarks.landmark[5].x * frame.shape[1])
                face_width = x_max - x_min
                box_width = int(face_width / 4)  # 얼굴의 절반 크기로 설정
                left_face_box = (int(nose_x - box_width), int(y_min), int(nose_x), int(y_max))
                right_face_box = (int(nose_x), int(y_min), int(nose_x + box_width), int(y_max))

                # 얼굴의 왼쪽 볼을 빨간색 원형으로 표시
                left_cheek_color = (0, 0, 255)  # 빨간색
                left_cheek_center = ((x_min + nose_x) // 2, (y_min + y_max) // 2)
                left_cheek_radius = min((nose_x - x_min) // 2, (y_max - y_min) // 2)
                cv2.circle(frame, (int(left_cheek_center[0]), int(left_cheek_center[1])), int(left_cheek_radius), left_cheek_color, 2)

                # 얼굴의 오른쪽 볼을 파란색 원형으로 표시
                right_cheek_color = (255, 0, 0)  # 파란색
                right_cheek_center = ((nose_x + x_max) // 2, (y_min + y_max) // 2)
                right_cheek_radius = min((x_max - nose_x) // 2, (y_max - y_min) // 2)
                cv2.circle(frame, (int(right_cheek_center[0]), int(right_cheek_center[1])), int(right_cheek_radius), right_cheek_color, 2)

                # 원 중심에서 수평선 그리기
                cv2.line(frame, (int(left_cheek_center[0] - left_cheek_radius), int(left_cheek_center[1])),
                         (int(right_cheek_center[0] + right_cheek_radius), int(right_cheek_center[1])),
                         (0, 255, 255), 2)
        
        # 결과 보여주기
        cv2.imshow('Face Landmarks', frame)
        
        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 작업 완료 후 해제
cap.release()
cv2.destroyAllWindows()
