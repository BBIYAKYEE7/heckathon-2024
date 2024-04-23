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
baseline_error = 10


baseline_set = False


prev_face_center = None
prev_face_angle = 0

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
                
                if not baseline_set:
                    baseline_length = nose_y
                    baseline_set = True
                
       
                nose_error = abs(nose_y - baseline_length)
                
              
                if nose_error > baseline_error:
                    baseline_length = nose_y
                
                
                cv2.line(frame, (0, baseline_length), (frame.shape[1], baseline_length), (255, 0, 0), 2)

                
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
                   
                    nose_movement = nose_y - baseline_length
                    
                    # 스크롤 제어
                    if nose_movement < 0:
                        pyautogui.scroll(-scroll_speed)  
                    elif nose_movement > 0:
                        pyautogui.scroll(scroll_speed) 
                
              
                prev_face_center = face_center
                prev_face_angle = face_angle
                
             
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
        
        # 결과 보여주기
        cv2.imshow('Face Landmarks', frame)
        
        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 작업 완료 후 해제
cap.release()
cv2.destroyAllWindows()
