import cv2
import mediapipe as mp
import math
import webbrowser

# 두 점 사이의 거리를 계산하는 함수
def calculate_distance(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# 길이를 화면에 표시하는 함수
def display_length(frame, length, position):
    cv2.putText(frame, f'Length: {length:.2f} cm', position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

def detect_hand():
    # 미디어 파이프 손 모델 로드
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    # 웹캠에서 영상을 가져옵니다.
    cap = cv2.VideoCapture(0)
    
    is_chrome_running = False  # 크롬 실행 여부를 나타내는 변수
    

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("비디오를 가져올 수 없습니다.")
            break

        # 이미지를 BGR에서 RGB로 변환하고 손 감지 수행
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 손 감지된 지점을 원으로 그립니다.
                for idx, point in enumerate(hand_landmarks.landmark):
                    x, y = int(point.x * frame.shape[1]), int(point.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # 검지와 엄지 손가락 끝점을 파란색 선으로 이어줍니다.
                index_tip_point = (int(hand_landmarks.landmark[8].x * frame.shape[1]), int(hand_landmarks.landmark[8].y * frame.shape[0]))
                thumb_tip_point = (int(hand_landmarks.landmark[4].x * frame.shape[1]), int(hand_landmarks.landmark[4].y * frame.shape[0]))
                cv2.line(frame, index_tip_point, thumb_tip_point, (255, 0, 0), 2)
                # 검지와 엄지 파란색 선의 길이 계산
                index_thumb_length = calculate_distance(index_tip_point, thumb_tip_point)
                # 검지와 엄지 파란색 선의 길이 표시
                display_length(frame, index_thumb_length, (index_tip_point[0] + 10, index_tip_point[1] + 10))

                # 새끼손가락과 손목 끝을 파란색 선으로 연결
                pinky_tip_point = (int(hand_landmarks.landmark[20].x * frame.shape[1]), int(hand_landmarks.landmark[20].y * frame.shape[0]))
                wrist_point = (int(hand_landmarks.landmark[0].x * frame.shape[1]), int(hand_landmarks.landmark[0].y * frame.shape[0]))
                cv2.line(frame, pinky_tip_point, wrist_point, (255, 0, 0), 2)
                # 새끼손가락과 손목 파란색 선의 길이 계산
                pinky_wrist_length = calculate_distance(pinky_tip_point, wrist_point)
                # 새끼손가락과 손목 파란색 선의 길이 표시
                display_length(frame, pinky_wrist_length, (pinky_tip_point[0] + 10, pinky_tip_point[1] + 10))

                # 길이가 30cm인 경우 크롬 실행
                if 30 <= index_thumb_length <= 38 and not is_chrome_running:
                    webbrowser.open("https://www.google.com", new=2)
                    is_chrome_running = True


        # 결과 표시
        cv2.imshow('Hand Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_hand()
