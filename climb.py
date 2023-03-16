import mediapipe as mp
from ultralytics import YOLO
import cv2


def coords(x, y):
    return (int(x * vid_width), int(y * vid_height))


def insideBox(p, boxes, allowance=15):

    for b in boxes:
        if (
            p[0] >= b[0] - allowance
            and p[1] >= b[1] - allowance
            and p[0] <= b[2] + allowance
            and p[1] <= b[3] + allowance
        ):
            return True
    return False


def drawLines(landmarks, boxes, checkHolds=False):

    if landmarks:

        l_wrist = coords(landmarks.landmark[15].x, landmarks.landmark[15].y)
        r_wrist = coords(landmarks.landmark[16].x, landmarks.landmark[16].y)

        l_ankle = coords(landmarks.landmark[27].x, landmarks.landmark[27].y)
        r_ankle = coords(landmarks.landmark[28].x, landmarks.landmark[28].y)

        m_shldr = coords(
            (landmarks.landmark[11].x + landmarks.landmark[12].x) / 2,
            (landmarks.landmark[11].y + landmarks.landmark[12].y) / 2,
        )

        m_hip = coords(
            (landmarks.landmark[23].x + landmarks.landmark[24].x) / 2,
            (landmarks.landmark[23].y + landmarks.landmark[24].y) / 2,
        )

        if checkHolds:
            
            lw_in = insideBox(l_wrist, boxes)
            rw_in = insideBox(r_wrist, boxes)
            la_in = insideBox(l_ankle, boxes)
            ra_in = insideBox(r_ankle, boxes)

            if lw_in :
                cv2.line(image, l_wrist, m_hip, (0, 255, 0), 1)
                if la_in :
                    cv2.line(image, l_wrist, l_ankle, (255, 0, 0), 2)
                if ra_in :
                    cv2.line(image, l_wrist, r_ankle, (255, 0, 0), 2)
            if rw_in :
                cv2.line(image, r_wrist, m_hip, (0, 255, 0), 1)
                if la_in :
                    cv2.line(image, r_wrist, l_ankle, (0, 0, 255), 2)
                if ra_in :
                    cv2.line(image, r_wrist, r_ankle, (0, 0, 255), 2)
            if lw_in and rw_in :
                cv2.line(image, l_wrist, r_wrist, (255, 0, 255), 1)
            if la_in :
                cv2.line(image, r_ankle, m_shldr, (255, 255, 0), 1)
            if ra_in :
                cv2.line(image, r_ankle, m_shldr, (255, 255, 0), 1)
            if la_in and ra_in :
                cv2.line(image, l_ankle, r_ankle, (255, 0, 255), 1)
            
        else :

            cv2.line(image, l_wrist, l_ankle, (255, 0, 0), 2)
            cv2.line(image, l_wrist, r_ankle, (255, 0, 0), 2)
            cv2.line(image, l_wrist, m_hip, (0, 255, 0), 1)
            cv2.line(image, r_wrist, l_ankle, (0, 0, 255), 2)
            cv2.line(image, r_wrist, r_ankle, (0, 0, 255), 2)
            cv2.line(image, r_wrist, m_hip, (0, 255, 0), 1)
            cv2.line(image, l_ankle, m_shldr, (255, 255, 0), 1)
            cv2.line(image, r_ankle, m_shldr, (255, 255, 0), 1)
            cv2.line(image, l_wrist, r_wrist, (255, 0, 255), 1)
            cv2.line(image, l_ankle, r_ankle, (255, 0, 255), 1)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

model = YOLO("best.pt")
model.fuse()

cap = cv2.VideoCapture("boulder.mp4")
vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while cap.isOpened():

        success, image = cap.read()

        if not success:
            break

        holds = model.predict(image)
        boxes = holds[0].boxes.xyxy

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        drawLines(results.pose_landmarks, boxes, False)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("", image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
