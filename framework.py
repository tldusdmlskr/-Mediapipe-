# ================================================
# 시연 2: MediaPipe Framework 직접 사용
# 새 API(mp.tasks) 기반으로
# 커스텀 파이프라인을 직접 구성하는 방식
# ================================================

import cv2
import mediapipe as mp
import math

# ── Framework 핵심 컴포넌트 직접 import ──────────
# Solutions처럼 mp.solutions.xxx 가 아니라
# 내부 tasks 모듈을 직접 불러옴
BaseOptions           = mp.tasks.BaseOptions
PoseLandmarker        = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

MODEL_PATH = "pose_landmarker_full.task"

# ============================================================
# [핵심 개념 설명]
# Solutions 방식:  33개 랜드마크 전체 반환 → 내부 로직 숨겨짐
# Framework 방식:  Options로 파이프라인 직접 구성
#                  → 관심 랜드마크만 필터링
#                  → visibility 기준 직접 설정
#                  → 연결선/색상/렌더링 직접 정의
# ============================================================

class CustomPoseFramework:
    """
    MediaPipe Framework를 직접 제어하는 커스텀 클래스.

    Solutions와 달리:
    - 옵션으로 파이프라인 구성을 직접 제어
    - 33개 중 관심 랜드마크만 선택적으로 처리
    - visibility 기반 커스텀 필터링 로직 적용
    - 연결선/색상/렌더링을 직접 정의
    """

    # 관심 랜드마크만 선택 (상체 + 무릎까지)
    UPPER_BODY_IDS = {
        0:  "코",
        11: "왼쪽 어깨",
        12: "오른쪽 어깨",
        13: "왼쪽 팔꿈치",
        14: "오른쪽 팔꿈치",
        15: "왼쪽 손목",
        16: "오른쪽 손목",
        23: "왼쪽 엉덩이",
        24: "오른쪽 엉덩이",
        25: "왼쪽 무릎",
        26: "오른쪽 무릎",
    }

    # 커스텀 연결선 정의 (Solutions의 POSE_CONNECTIONS 대신 직접 정의)
    CUSTOM_CONNECTIONS = [
        (11, 12),             # 어깨
        (11, 13), (13, 15),  # 왼팔
        (12, 14), (14, 16),  # 오른팔
        (11, 23), (12, 24),  # 몸통
        (23, 24),             # 골반
        (23, 25), (24, 26),  # 허벅지
    ]

    def __init__(self, model_complexity="full", min_confidence=0.5):
        # ── Framework 레벨 옵션 직접 구성 ────────
        # Solutions는 이 과정이 내부에 숨겨져 있음
        self._options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_poses=1,                              # 감지할 사람 수 직접 지정
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            output_segmentation_masks=False           # 불필요한 출력 스트림 비활성화
        )
        self._landmarker = PoseLandmarker.create_from_options(self._options)

    # ── Calculator 1: 패킷 처리 (추론) ───────────
    def process_packet(self, frame_rgb, timestamp_ms):
        """
        커스텀 Calculator의 Process() 메서드에 해당.
        입력 Packet(frame) → 추론 → 출력 Packet(landmarks)
        """
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    # ── Calculator 2: 커스텀 필터링 ──────────────
    def filter_landmarks(self, result, h, w):
        """
        33개 전체 랜드마크 중 관심 있는 것만 추출.
        Solutions에서는 이 과정이 내부에 숨겨져 있음.
        """
        if not result.pose_landmarks:
            return {}

        landmarks = result.pose_landmarks[0]  # 첫 번째 사람
        filtered = {}

        for idx, name in self.UPPER_BODY_IDS.items():
            lm = landmarks[idx]
            # visibility: 해당 랜드마크가 보이는 정도 (0~1)
            # 직접 기준값 설정 → Solutions는 이걸 내부에서 처리
            if lm.visibility > 0.5:
                filtered[idx] = {
                    "name":       name,
                    "x":          int(lm.x * w),
                    "y":          int(lm.y * h),
                    "z":          round(lm.z, 4),  # 깊이 정보
                    "visibility": round(lm.visibility, 2)
                }
        return filtered

    # ── Calculator 3: 커스텀 렌더링 ──────────────
    def draw_custom(self, frame, filtered):
        """
        직접 정의한 연결선과 색상으로 렌더링.
        Solutions의 draw_landmarks() 대신 직접 구현.
        """
        if not filtered:
            return frame

        # 커스텀 연결선 그리기
        for (id_a, id_b) in self.CUSTOM_CONNECTIONS:
            if id_a in filtered and id_b in filtered:
                pt_a = (filtered[id_a]["x"], filtered[id_a]["y"])
                pt_b = (filtered[id_b]["x"], filtered[id_b]["y"])
                cv2.line(frame, pt_a, pt_b, (100, 220, 100), 2)

        # 랜드마크 점 + 이름 그리기
        for idx, info in filtered.items():
            cx, cy = info["x"], info["y"]
            # visibility에 따라 색 변화 (낮으면 빨강, 높으면 초록)
            vis   = info["visibility"]
            color = (0, int(255 * vis), int(255 * (1 - vis)))
            cv2.circle(frame, (cx, cy), 6, color, -1)
            cv2.putText(frame, info["name"],
                        (cx + 8, cy - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame

    def close(self):
        self._landmarker.close()


# ── 메인 루프 ─────────────────────────────────────
def main():
    pipeline = CustomPoseFramework(min_confidence=0.5)
    cap      = cv2.VideoCapture(0)

    print("▶ Framework 직접 제어 모드")
    print("  - 관심 랜드마크만 필터링 (33개 → 11개)")
    print("  - visibility 기반 색상 표시")
    print("  - 'q' 키로 종료\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _      = frame.shape
        frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC) * 1000)

        # ── Calculator 1: 포즈 추론 (패킷 처리) ──
        result = pipeline.process_packet(frame_rgb, timestamp_ms)

        # ── Calculator 2: 커스텀 필터링 ──────────
        filtered = pipeline.filter_landmarks(result, h, w)

        # ── Calculator 3: 커스텀 렌더링 ──────────
        frame = pipeline.draw_custom(frame, filtered)

        # 감지 현황 표시
        cv2.putText(frame,
                    f"감지된 포인트: {len(filtered)} / {len(pipeline.UPPER_BODY_IDS)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("MediaPipe Framework - Custom Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()