import cv2
import mediapipe as mp
import csv
import os

camera = cv2.VideoCapture('/Users/santiagorojasjaramillo/Desktop/Prueba_1234/fist_l.mp4')
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# properties
n_frames = camera.get(cv2.CAP_PROP_FRAME_COUNT)
fps = camera.get(cv2.CAP_PROP_FPS)
width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(n_frames, fps, width, height)

if not os.path.exists('/Users/santiagorojasjaramillo/Desktop/mov_csv'):
    os.mkdir('/Users/santiagorojasjaramillo/Desktop/mov_csv')

ret = True
frame = 0
while ret:
    ret, image = camera.read()
    if ret:
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_hand_landmarks:
                continue
            #row = [camera.get(cv2.CAP_PROP_POS_MSEC)]
            row = [float(frame)/fps]
            for finger in results.multi_hand_landmarks[0].landmark:
                row.append(finger.x)
                row.append(finger.y)
                row.append(finger.z)

            with open('/Users/santiagorojasjaramillo/Desktop/mov_csv/fist_l_parkinson.csv', mode='a') as f:
                f_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                f_writer.writerow(row)

        frame += 1

