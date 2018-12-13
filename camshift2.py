import numpy as np
import cv2

cap = cv2.VideoCapture('test_videos/street.mp4')

# 获取视频的第一个帧
ret, frame = cap.read()

# 设置窗口的初始位置
x, y, w, h = 192, 206, 18, 120
track_window = (x, y, w, h)

# 设置跟踪的区域roi
roi = frame[y:y + h, x:x + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32,)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# 设置迭代次数，或者迭代10次或者至少移动1次
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    # print(type(frame))
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # 对新的位置进行meanShift操作
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # 在图上进行绘制
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img = cv2.polylines(frame, [pts], True, 255, 2)
        cv2.imshow("img", img)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
