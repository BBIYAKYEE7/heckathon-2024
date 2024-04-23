import cv2
import numpy as np
import pyautogui
import time
import math
import mediapipe as mp
import winsound  

# Mediapipe 모듈 초기화
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 스크롤 속도 및 임계값 설정
scroll_speed = 5
scroll_threshold = 0.05  

# 초기 설정
baseline_length = None
baseline_error = 23
baseline_set = False

initial_baseline_length = None
initial_baseline_set = False

keyboard = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]
]

keyboard_displayed = False

# 키보드 그리기 함수
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

                cv2.rectangle(img, (key_x, key_y), (key_x + key_width, key_y + key_height), (255, 255, 255), 1)
                cv2.putText(img, key, (key_x + 10, key_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                center_x = key_x + key_width // 2
                center_y = key_y + key_height // 2
                cv2.circle(img, (center_x, center_y), 3, (255, 0, 0), cv2.FILLED)
        
        cv2.imshow("Hand Tracking", img)
    else:
        pass

# 손 감지 클래스
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
    
    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
 
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.circle(img, (x1, y1), 6, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 6, (0, 255, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 6, (0, 255, 255), cv2.FILLED)
 
        length = math.hypot(x2 - x1, y2 - y1)
 
        return length, img, [x1, y1, x2, y2, cx, cy]
 
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
            cv2.circle(img, (x1, y1), 6, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 6, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 6, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
 
        return angle

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def display_length(img, length, position):
    cv2.putText(img, f"Length: {int(length)}", position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

def calculate_camera_hand_distance(lmList):
    if len(lmList) >= 21:
        thumb_tip = lmList[4]
        index_finger_tip = lmList[8]
        distance = math.sqrt((thumb_tip[1] - index_finger_tip[1])**2 + (thumb_tip[2] - index_finger_tip[2])**2)
        return distance
    else:
        return None
    
move_threshold = 10

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


cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.75)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    if lmList:
        for landmark in lmList:
            id, x, y = landmark
            cv2.circle(img, (x, y), 6, (255, 0, 255), cv2.FILLED)
            cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        angle = detector.findAngle(0, 12, 8, img, draw=False)

        index_tip_point = (lmList[8][1], lmList[8][2])
        thumb_tip_point = (lmList[4][1], lmList[4][2])
        cv2.line(img, index_tip_point, thumb_tip_point, (255, 0, 0), 2)
        index_thumb_length = calculate_distance(index_tip_point, thumb_tip_point)
        display_length(img, index_thumb_length, (index_tip_point[0] + 10, index_tip_point[1] + 10))

        pinky_tip_point = (lmList[20][1], lmList[20][2])
        wrist_point = (lmList[0][1], lmList[0][2])
        cv2.line(img, pinky_tip_point, wrist_point, (255, 0, 0), 2)
        pinky_wrist_length = calculate_distance(pinky_tip_point, wrist_point)
        display_length(img, pinky_wrist_length, (pinky_tip_point[0] + 10, pinky_tip_point[1] + 10))
        
        index_tip_point = (lmList[8][1], lmList[8][2])
        middle_finger_tip = (lmList[12][1], lmList[12][2])
        cv2.line(img, index_tip_point, middle_finger_tip, (255, 0, 0), 2)
        index_middle_distance = calculate_distance(index_tip_point, middle_finger_tip)    
        display_length(img, index_middle_distance, (middle_finger_tip[0] + 10, middle_finger_tip[1] + 10))
        
        if 17 <= index_middle_distance <= 45:
            pyautogui.click(button='left')

        index_finger_joint = (lmList[16][1], lmList[16][2])
        middle_finger_joint = (lmList[12][1], lmList[12][2])
        cv2.line(img, index_finger_joint, middle_finger_joint, (255, 0, 0), 2)
        midpoint_x = (index_finger_joint[0] + middle_finger_joint[0]) // 2
        midpoint_y = (index_finger_joint[1] + middle_finger_joint[1]) // 2
        cv2.circle(img, (midpoint_x, midpoint_y), 6, (255, 255, 0), cv2.FILLED)

        index_finger_tip = (lmList[1][1], lmList[1][2])
        index_finger_midpoint = (midpoint_x, midpoint_y)
        cv2.line(img, index_finger_tip, index_finger_midpoint, (255, 255, 0), 2)
        index_finger_midpoint_length = calculate_distance(index_finger_tip, index_finger_midpoint)
        display_length(img, index_finger_midpoint_length, (index_finger_midpoint[0] + 10, index_finger_midpoint[1] + 10))

        middle_finger_tip = (lmList[12][1], lmList[12][2])
        thumb_tip_point = (lmList[4][1], lmList[4][2])
        cv2.line(img, middle_finger_tip, thumb_tip_point, (255, 0, 0), 2)
        middle_thumb_length = calculate_distance(middle_finger_tip, thumb_tip_point)
        display_length(img, middle_thumb_length, (middle_finger_tip[0] + 10, middle_finger_tip[1] + 10))

        index_tip_x = lmList[8][1]
        index_tip_y = lmList[8][2]
        pyautogui.moveTo(index_tip_x, index_tip_y)

        if lmList:
            camera_hand_distance = calculate_camera_hand_distance(lmList)
            if camera_hand_distance:
                if camera_hand_distance < 0.4:
                    color = (0, 0, 255) 
                    message = "INCORRECT DISTANCE : {:.2f} m.".format(camera_hand_distance)
                    winsound.Beep(1000, 500)
                elif camera_hand_distance >= 0.4 and camera_hand_distance < 0.6:
                    color = (255, 0, 0)  
                    message = "CORRECT DISTANCE : {:.2f} m.".format(camera_hand_distance)
                else:
                    color = (0, 255, 0)  
                    message = "excellent : {:.2f}m.".format(camera_hand_distance)
        else:
            message = ""
        
        draw_keyboard(img)    
        

        # 거리 표시
        cv2.putText(
            img,      
            message,
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

    # 얼굴 메시 처리
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    nose_center = None
    prev_nose_center = None

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 얼굴 주위에 사각형 그리기
            x_min = min(int(coord.x * img.shape[1]) for coord in face_landmarks.landmark)
            x_max = max(int(coord.x * img.shape[1]) for coord in face_landmarks.landmark)
            y_min = min(int(coord.y * img.shape[0]) for coord in face_landmarks.landmark)
            y_max = max(int(coord.y * img.shape[0]) for coord in face_landmarks.landmark)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            nose_center = find_nose_center(face_landmarks)
            
    if prev_nose_center is not None and nose_center is not None:
        distance = calculate_distance(prev_nose_center, nose_center)
        if distance > move_threshold:
            nose_center = prev_nose_center        

    if nose_center is not None:
        horizon_y = nose_center[1] - 10
        cv2.line(img, (0, horizon_y), (img.shape[1], horizon_y), (0, 225, 255), 2)
        if horizon_y > 0:  # 코의 위치가 수평선 아래에 있으면
            pyautogui.scroll(-scroll_speed)  # 위로 스크롤
        elif horizon_y < 0:
            pyautogui.scroll(scroll_speed)  # 아래로 스크롤

  
    # 화면 표시
    cv2.imshow("Hand Tracking", img)
    
    # 종료 조건
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
