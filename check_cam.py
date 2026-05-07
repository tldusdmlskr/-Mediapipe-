import cv2

cap = cv2.VideoCapture(0)  # 0: 기본 웹캠
if cap.isOpened():
    print("웹캠 연결 OK!")
else:
    print("웹캠 연결 실패 - 번호를 1이나 2로 바꿔보세요")
cap.release()