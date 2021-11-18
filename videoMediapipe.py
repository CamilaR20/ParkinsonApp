import cv2
import mediapipe as mp
import csv

if __name__ == '__main__':
    camera = cv2.VideoCapture('/Users/camilaroa/Downloads/ParkinsonVideos/0009/25-10-2021, 14-00, OFF/pronosup_r.mp4')
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # properties
    n_frames = camera.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = camera.get(cv2.CAP_PROP_FPS)
    width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(n_frames, fps, width, height)

    ret = True
    while ret:
        ret, image = camera.read()
        if ret:
            with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if not results.multi_hand_landmarks:
                    print("no")
                    continue
                annotated_image = image.copy()
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.imshow("image", annotated_image)
                    cv2.waitKey(int(1000 / fps))