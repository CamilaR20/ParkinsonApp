import cv2
import mediapipe as mp

if __name__ == '__main__':
    image = cv2.imread("/Users/camilaroa/Downloads/ParkinsonVideos/0004/07-10-2021, 09-47, ON/fingertap_l.jpg")

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands



    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # print('Handedness:', results.multi_handedness)
        annotated_image = image.copy()
        for hand_landmarks in results.multi_hand_landmarks:
            print(
                f'Calibration coordinates: (',
                f'{hand_landmarks.landmark[0].y},'
                f'{hand_landmarks.landmark[9].y})'
            )
            distance = hand_landmarks.landmark[0].y - hand_landmarks.landmark[9].y
            print(distance)

            mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.imshow("image", annotated_image)
            cv2.waitKey(0)

            print(image.shape)
