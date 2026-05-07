import cv2
import mediapipe as mp
import math

BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

MODEL_PATH = "pose_landmarker_full.task"

def calc_angle(a, b, c):
    ab = (a[0]-b[0], a[1]-b[1])
    cb = (c[0]-b[0], c[1]-b[1])
    dot    = ab[0]*cb[0] + ab[1]*cb[1]
    mag_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    mag_cb = math.sqrt(cb[0]**2 + cb[1]**2)
    if mag_ab * mag_cb == 0:
        return 0.0
    return round(math.degrees(math.acos(
        max(-1, min(1, dot / (mag_ab * mag_cb)))
    )), 1)

CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28)
]

def draw_landmarks_on_frame(frame, detection_result):
    if not detection_result.pose_landmarks:
        return frame

    h, w   = frame.shape[:2]
    landmarks = detection_result.pose_landmarks[0]

    # 연결선 그리기
    for (a, b) in CONNECTIONS:
        sx, sy = int(landmarks[a].x * w), int(landmarks[a].y * h)
        ex, ey = int(landmarks[b].x * w), int(landmarks[b].y * h)
        cv2.line(frame, (sx, sy), (ex, ey), (255, 255, 255), 2)

    # 랜드마크 점 그리기
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

    # 팔꿈치 각도 계산 및 표시
    def get_pt(idx):
        return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

    r_elbow = get_pt(14)
    l_elbow = get_pt(13)
    angle_r = calc_angle(get_pt(12), r_elbow, get_pt(16))
    angle_l = calc_angle(get_pt(11), l_elbow, get_pt(15))

    cv2.putText(frame, f"R elbow: {angle_r}",
                (r_elbow[0]+10, r_elbow[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(frame, f"L elbow: {angle_l}",
                (l_elbow[0]+10, l_elbow[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    return frame

# 옵션 설정
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO
)

cap = cv2.VideoCapture(0)
print("▶ 'q' 키로 종료")

with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC) * 1000)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        frame  = draw_landmarks_on_frame(frame, result)

        cv2.imshow("MediaPipe Pose - Solutions", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()