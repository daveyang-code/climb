import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def coords(x, y):
    return (int(x * vid_width), int(y * vid_height))


def drawLines(landmarks):
    if landmarks:

        l_wrist = coords(landmarks.landmark[15].x, landmarks.landmark[15].y)
        r_wrist = coords(landmarks.landmark[16].x, landmarks.landmark[16].y)

        l_ankle = coords(landmarks.landmark[27].x, landmarks.landmark[27].y)
        r_ankle = coords(landmarks.landmark[28].x, landmarks.landmark[28].y)

        cv2.line(image, l_wrist, l_ankle, (255, 0, 0), 2)
        cv2.line(image, l_wrist, r_ankle, (255, 0, 0), 2)
        cv2.line(image, r_wrist, l_ankle, (0, 0, 255), 2)
        cv2.line(image, r_wrist, r_ankle, (0, 0, 255), 2)
        cv2.line(image, l_wrist, r_wrist, (255, 0, 255), 2)
        cv2.line(image, l_ankle, r_ankle, (255, 0, 255), 2)

cap = cv2.VideoCapture("boulder.mp4")
vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():

        success, image = cap.read()

        if not success:
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        drawLines(results.pose_landmarks)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
        cv2.imshow("", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
