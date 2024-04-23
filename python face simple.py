import cv2
import mediapipe as mp
import math

# 미디어파이프 얼굴 랜드마크 모듈 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 이전 프레임의 눈 중심 좌표 초기화
prev_left_eye_center = None
prev_right_eye_center = None

# 스크롤 제어를 위한 변수 초기화
scroll_speed = 1  # 스크롤 속도
scroll_threshold = 0.1  # 스크롤 감지 임계값

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        # 웹캠에서 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break
        
        # OpenCV의 BGR 이미지를 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 얼굴 랜드마크 검출 수행
        results = face_mesh.process(rgb_frame)
        
        # 얼굴 랜드마크가 있을 경우
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 눈 주변 랜드마크 인덱스
                left_eye_landmark_indices = [33, 246, 161, 160, 159, 158, 157]
                right_eye_landmark_indices = [362, 398, 384, 385, 386, 387, 388]
                
                # 왼쪽 눈의 중심 좌표 계산
                left_eye_points = [face_landmarks.landmark[idx] for idx in left_eye_landmark_indices]
                left_eye_center_x = sum(point.x for point in left_eye_points) / len(left_eye_points)
                left_eye_center_y = sum(point.y for point in left_eye_points) / len(left_eye_points)
                left_eye_center = (int(left_eye_center_x * frame.shape[1]), int(left_eye_center_y * frame.shape[0]))
                
                # 오른쪽 눈의 중심 좌표 계산
                right_eye_points = [face_landmarks.landmark[idx] for idx in right_eye_landmark_indices]
                right_eye_center_x = sum(point.x for point in right_eye_points) / len(right_eye_points)
                right_eye_center_y = sum(point.y for point in right_eye_points) / len(right_eye_points)
                right_eye_center = (int(right_eye_center_x * frame.shape[1]), int(right_eye_center_y * frame.shape[0]))
                
                # 이전 프레임이 있을 경우, 눈의 이동 방향 계산 및 시각화
                if prev_left_eye_center is not None and prev_right_eye_center is not None:
                    # 왼쪽 눈의 이동 방향 계산
                    left_eye_direction = (left_eye_center[0] - prev_left_eye_center[0], left_eye_center[1] - prev_left_eye_center[1])
                    left_eye_direction_length = math.sqrt(left_eye_direction[0] ** 2 + left_eye_direction[1] ** 2)
                    if left_eye_direction_length != 0:  # 방향 벡터의 길이가 0인 경우 제외
                        left_eye_direction = (left_eye_direction[0] / left_eye_direction_length * scroll_speed, left_eye_direction[1] / left_eye_direction_length * scroll_speed)
                        if left_eye_direction[1] > scroll_threshold:
                            # 아래로 스크롤
                            print("Scroll down")
                        elif left_eye_direction[1] < -scroll_threshold:
                            # 위로 스크롤
                            print("Scroll up")
                    
                    # 오른쪽 눈의 이동 방향 계산
                    right_eye_direction = (right_eye_center[0] - prev_right_eye_center[0], right_eye_center[1] - prev_right_eye_center[1])
                    right_eye_direction_length = math.sqrt(right_eye_direction[0] ** 2 + right_eye_direction[1] ** 2)
                    if right_eye_direction_length != 0:  # 방향 벡터의 길이가 0인 경우 제외
                        right_eye_direction = (right_eye_direction[0] / right_eye_direction_length * scroll_speed, right_eye_direction[1] / right_eye_direction_length * scroll_speed)
                        if right_eye_direction[1] > scroll_threshold:
                            # 아래로 스크롤
                            print("Scroll down")
                        elif right_eye_direction[1] < -scroll_threshold:
                            # 위로 스크롤
                            print("Scroll up")
                
                # 현재 눈 중심 좌표를 이전 프레임의 눈 중심 좌표로 설정
                prev_left_eye_center = left_eye_center
                prev_right_eye_center = right_eye_center
                
                # 눈 중심 좌표를 이용하여 동공 위치 추정
                cv2.circle(frame, left_eye_center, 5, (0, 255, 0), -1)
                cv2.circle(frame, right_eye_center, 5, (0, 255, 0), -1)
        
        # 결과 보여주기
        cv2.imshow('Face Landmarks', frame)
        
        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 작업 완료 후 해제
cap.release()
cv2.destroyAllWindows()
