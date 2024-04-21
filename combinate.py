import cv2
import numpy as np
import pyautogui
import time
import math
import mediapipe as mp
import winsound  

#all code developed by Minho Choi 

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

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

# Define the keyboard layout globally
keyboard = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]
]

# 키보드 화면을 표시하는지 여부를 나타내는 변수
keyboard_displayed = False

# 키보드 자판을 그리는 함수
def draw_keyboard(img, show=True):
    if show:
        key_width = 40
        key_height = 40
        key_padding = 10
        start_x = 80
        start_y = 300
        
        for row_index, row in enumerate(keyboard):
            for col_index, key in enumerate(row):
                key_x = start_x + col_index * (key_width + key_padding)
                key_y = start_y + row_index * (key_height + key_padding)
                
                # Draw key rectangle
                cv2.rectangle(img, (key_x, key_y), (key_x + key_width, key_y + key_height), (255, 255, 255), 1)
                
                # Draw key label
                cv2.putText(img, key, (key_x + 10, key_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Draw blue dot in the center of each key
                center_x = key_x + key_width // 2
                center_y = key_y + key_height // 2
                cv2.circle(img, (center_x, center_y), 3, (255, 0, 0), cv2.FILLED)
    else:
        pass

class HandDetector:
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
        self.lmList = []
        self.results = None
        self.mode = mode
        self.maxHands = maxHands
        self.modelComplex = modelComplexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.direction = None
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]
 
    def findHands(self, img, draw=True):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_img)
 
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
 
        return img
 
    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmList = []
 
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
 
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
 
                if draw:
                    cv2.circle(img, (cx, cy), 6, (0, 0, 255), cv2.FILLED)
 
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
 
            if draw:
                cv2.rectangle(img, (bbox[0]-20, bbox[1]-20), (bbox[2]+20, bbox[3]+20), (0, 255, 0), 2)
 
        return self.lmList, bbox
 
    def fingersUp(self):
        fingers = []
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
 
        return fingers
 
    def findAngle(self, p1, p2, p3, img, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
 
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), cv2.FILLED)
 
        return angle

def calculate_distance(point1, point2):
    return int(math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2))

def display_length(img, length, position):
    cv2.putText(img, f"Length: {length}", position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def calculate_camera_hand_distance(lmList):
    if len(lmList) > 0:
        x_min, y_min = lmList[0][1], lmList[0][2]
        x_max, y_max = lmList[0][1], lmList[0][2]

        for lm in lmList:
            x, y = lm[1], lm[2]
            if x < x_min:
                x_min = x
            elif x > x_max:
                x_max = x
            if y < y_min:
                y_min = y
            elif y > y_max:
                y_max = y

        bbox_width = x_max - x_min
        bbox_height = y_max - y_min

        distance = 100 / max(bbox_width, bbox_height)  # 거리 100 최대로.

        return distance
    else:
        return None

move_threshold = 10

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = HandDetector()

# "Hand Tracking" 창 생성
cv2.namedWindow("Hand Tracking")

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
 
    if lmList:
        for landmark in lmList:
            id, x, y = landmark
            cv2.circle(img, (x, y), 6, (255, 0, 255), cv2.FILLED)
            cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 손의 기울기/각도
        angle = detector.findAngle(0, 12, 8, img, draw=False)

        # 검지와 엄지 손가락 끝을 연결하고 길이 계산 및 표시
        index_tip_point = (lmList[8][1], lmList[8][2])
        thumb_tip_point = (lmList[4][1], lmList[4][2])
        cv2.line(img, index_tip_point, thumb_tip_point, (255, 0, 0), 2)
        index_thumb_length = calculate_distance(index_tip_point, thumb_tip_point)
        display_length(img, index_thumb_length, (index_tip_point[0] + 10, index_tip_point[1] + 10))

        # 새끼손가락과 손목 끝을 연결하고 길이 계산 및 표시
        pinky_tip_point = (lmList[20][1], lmList[20][2])
        wrist_point = (lmList[0][1], lmList[0][2])
        cv2.line(img, pinky_tip_point, wrist_point, (255, 0, 0), 2)
        pinky_wrist_length = calculate_distance(pinky_tip_point, wrist_point)
        display_length(img, pinky_wrist_length, (pinky_tip_point[0] + 10, pinky_tip_point[1] + 10))
        
        # 8과 12 점을 연결하고 파란색으로 표시
        index_tip_point = (lmList[8][1], lmList[8][2])
        middle_finger_tip = (lmList[12][1], lmList[12][2])
        cv2.line(img, index_tip_point, middle_finger_tip, (255, 0, 0), 2)
        index_middle_distance = calculate_distance(index_tip_point, middle_finger_tip)
        display_length(img, index_middle_distance, (middle_finger_tip[0] + 10, middle_finger_tip[1] + 10))
        
        # 만약 8번과 12번 랜드마크 사이의 거리가 17-20 내에 있다면 왼쪽 클릭으로 작동
        if 17 <= index_middle_distance <= 45:
            pyautogui.click(button='left')
            

        # 16과 12 점을 연결하고 파란색으로 표시
        index_finger_joint = (lmList[16][1], lmList[16][2])
        middle_finger_joint = (lmList[12][1], lmList[12][2])
        cv2.line(img, index_finger_joint, middle_finger_joint, (255, 0, 0), 2)

        # 16-12 랜드마크에 연결된 파란색 선에 중앙에 점.
        midpoint_x = (index_finger_joint[0] + middle_finger_joint[0]) // 2
        midpoint_y = (index_finger_joint[1] + middle_finger_joint[1]) // 2
        cv2.circle(img, (midpoint_x, midpoint_y), 6, (255, 255, 0), cv2.FILLED)

        # 16-12중앙점과 1번 랜드마크 연결
        index_finger_tip = (lmList[1][1], lmList[1][2])
        index_finger_midpoint = (midpoint_x, midpoint_y)
        cv2.line(img, index_finger_tip, index_finger_midpoint, (255, 255, 0), 2)
        index_finger_midpoint_length = calculate_distance(index_finger_tip, index_finger_midpoint)
        display_length(img, index_finger_midpoint_length, (index_finger_midpoint[0] + 10, index_finger_midpoint[1] + 10))
        
        
        # 12번(가운데 손가락)과 4번(엄지 손가락)을 연결하고 길이 계산 및 표시
        middle_finger_tip = (lmList[12][1], lmList[12][2])
        thumb_tip_point = (lmList[4][1], lmList[4][2])
        cv2.line(img, middle_finger_tip, thumb_tip_point, (255, 0, 0), 2)
        middle_thumb_length = calculate_distance(middle_finger_tip, thumb_tip_point)
        display_length(img, middle_thumb_length, (middle_finger_tip[0] + 10, middle_finger_tip[1] + 10))


        
        index_tip_x = lmList[8][1]
        index_tip_y = lmList[8][2]  # y 좌표의 부호를 반대로 변경하여 커서가 반대 방향으로 움직이도록 함
        pyautogui.moveTo(index_tip_x, index_tip_y)


        if lmList:
            camera_hand_distance = calculate_camera_hand_distance(lmList)

            # 거리에 따라 색상 및 메시지 변경함
            if camera_hand_distance:
                if camera_hand_distance < 0.4:  # 60cm
                    color = (0, 0, 255)  # 빨간색
                    message = "INCORRECT DISTANCE : {:.2f} m.".format(camera_hand_distance)
                    # 경고음 재생
                    winsound.Beep(1000, 500)
                elif camera_hand_distance >= 0.4 and camera_hand_distance < 0.6:  # 60cm ~ 80cm
                    color = (255, 0, 0)  # 파란색
                    message = "CORRECT DISTANCE : {:.2f} m.".format(camera_hand_distance)
                else:
                    color = (0, 255, 0)  # 녹색
                    message = "excellent : {:.2f}m.".format(camera_hand_distance)
        else:
            message = ""

        # 코 위치 표시
        cv2.circle(img, (50, 50), 5, (0, 255, 255), cv2.FILLED)
        cv2.putText(img, "Nose", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 키보드 위치 표시
        draw_keyboard(img)

        # 손가락 길이와 키보드 거리를 비교하여 키보드를 누를지 결정
        if index_thumb_length >= 3 and camera_hand_distance is not None:
            if camera_hand_distance <= 0.4:
                pyautogui.press('space')

        # 거리 메시지 출력
        cv2.putText(
            img,
            message,
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
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

                # 원형 방향 표시
                left_cheek_direction = math.atan2((left_cheek_center[1] - prev_face_center[1]), (prev_face_center[0] - left_cheek_center[0]))
                right_cheek_direction = math.atan2((right_cheek_center[1] - prev_face_center[1]), (right_cheek_center[0] - prev_face_center[0]))
                
                # 원형 방향에 따라 방향 표시선을 그립니다.
                cv2.line(frame, left_cheek_center, (int(left_cheek_center[0] + left_cheek_radius * math.cos(left_cheek_direction)), int(left_cheek_center[1] - left_cheek_radius * math.sin(left_cheek_direction))), (0, 255, 255), 2)
                cv2.line(frame, right_cheek_center, (int(right_cheek_center[0] + right_cheek_radius * math.cos(right_cheek_direction)), int(right_cheek_center[1] - right_cheek_radius * math.sin(right_cheek_direction))), (0, 255, 255), 2)

                # 두 볼 사이의 거리를 계산하여 벌어짐 감지
                cheek_distance = calculate_distance(left_cheek_center, right_cheek_center)
                if baseline_length is None:
                    baseline_length = cheek_distance

                # 초기 코의 위치와 두 볼 사이의 거리를 비교하여 벌어짐 감지
                if baseline_length is not None:
                    if cheek_distance > baseline_length + scroll_threshold:  
                        pyautogui.scroll(-scroll_speed)  # 스크롤 아래로
                    elif cheek_distance < baseline_length - scroll_threshold:  
                        pyautogui.scroll(scroll_speed)  # 스크롤 위로
            prev_cheek_distance = cheek_distance
    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
