'''
多对象目标跟踪器
'''
import imutils
import numpy as np
import time
import cv2

# 视频文件路径和OpenCV对象跟踪器
video = "test_videos/street.mp4"
object_tracker = "kcf"

# 定义7种可用跟踪器
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}
# 初始化多对象跟踪器
trackers = cv2.MultiTracker_create()

# 初始化视频流
cap = cv2.VideoCapture(video)

# 循环帧并开始多目标跟踪
while True:
    # 获取当前视频的帧
    ret, frame = cap.read()

    # check to see if we have reached the end of the stream
    if frame is None:
        break
    # 将当前帧重置 (加快处理速度)
    frame = imutils.resize(frame, width=600)

    # 对于每一个被跟踪的对象矩形框进行更新
    (success, boxes) = trackers.update(frame)

    # 检查边界框并在帧上进行绘制
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        print(x,y,w,h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 显示框架以及选择要跟踪的对象
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(300) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROI("Frame", frame, fromCenter=False,
                            showCrosshair=True)

        # create a new object tracker for the bounding box and add it
        # to our multi-object tracker
        tracker = OPENCV_OBJECT_TRACKERS[object_tracker]()
        trackers.add(tracker, frame, box)
        # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

# 关闭所有窗口
cv2.destroyAllWindows()
cap.release()
