'''
基于卡尔曼滤波的Cam-Shift对象跟踪
'''
import numpy as np
import cv2

keep_processing = True
camera_to_use = 0;
selection_in_progress = False  # 支持感兴趣区域选择

# 使用鼠标来选择区域
boxes = []
current_mouse_position = np.ones(2, dtype=np.int32)


# 使用鼠标绘制矩形框选择区域
def on_mouse(event, x, y, flags, params):
    global boxes
    global selection_in_progress

    current_mouse_position[0] = x
    current_mouse_position[1] = y

    # 左击鼠标(按下)
    if event == cv2.EVENT_LBUTTONDOWN:
        boxes = []
        sbox = [x, y]
        selection_in_progress = True
        boxes.append(sbox)
    elif event == cv2.EVENT_LBUTTONUP:
        ebox = [x, y]
        selection_in_progress = False
        boxes.append(ebox)


# 返回所选择矩形框的中心点
def center(points):
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4.0
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4.0
    return np.array([np.float32(x), np.float32(y)], np.float32)


# 用作call-back当每次轨迹条移动时，可以当做什么也不做
def nothing(x):
    pass


# 定义视频捕捉对象
cap = cv2.VideoCapture(0)

# 定义展示的窗口名
windowName = "Kalman Object Tracking"
windowName2 = "Hue histogram back projection"
windowNameSelection = "initial selected region"

# 初始化卡尔曼滤波器所有参数

kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], np.float32)
measurement = np.array((2, 1), np.float32)
prediction = np.zeros((2, 1), np.float32)

# 打开视频
file = "test_videos/street.mp4"
cap = cv2.VideoCapture(file)

# 创建窗口
cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
cv2.namedWindow(windowName2, cv2.WINDOW_NORMAL)
cv2.namedWindow(windowNameSelection, cv2.WINDOW_NORMAL)

# 设置HSV选择阈值的滑块
s_lower = 60
cv2.createTrackbar("s lower", windowName2, s_lower, 255, nothing)
s_upper = 255;
cv2.createTrackbar("s upper", windowName2, s_upper, 255, nothing)
v_lower = 32;
cv2.createTrackbar("v lower", windowName2, v_lower, 255, nothing)
v_upper = 255;
cv2.createTrackbar("v upper", windowName2, v_upper, 255, nothing)

# 设置一个鼠标回调函数
cv2.setMouseCallback(windowName, on_mouse, 0)
cropped = False

# 设置迭代次数，或者迭代10次或者至少移动1次
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while keep_processing:
    #视频成功打开
    if cap.isOpened:
        ret,frame=cap.read()

    #开始一个计时器（测试处理以及展示所花时间）
    start_timer=cv2.getTickCount()

    #从滑动条中获得参数
    s_lower = cv2.getTrackbarPos("s lower", windowName2);
    s_upper = cv2.getTrackbarPos("s upper", windowName2);
    v_lower = cv2.getTrackbarPos("v lower", windowName2);
    v_upper = cv2.getTrackbarPos("v upper", windowName2);

    #通过鼠标选择区域并且进行展示
    if len(boxes)>1:
        crop=frame[boxes[0][1]:boxes[1][1],boxes[0][0]:boxes[1][0]].copy()#获得选择的区域图

        h,w,c=crop.shape
        if h>0 and w>0:
            cropped=True

            #将选择区域转换成HSV形式
            hsv_crop=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
            #选择所有色彩（0->180)
            mask=cv2.inRange(hsv_crop,np.array((0.,float(s_lower),float(v_lower))),np.array((180.,
                                            float(s_upper),float(v_upper))))
            #构造一个色调和饱和度直方图并将其归一化
            crop_hist=cv2.calcHist([hsv_crop],[0,1],mask,[180,255],[0,180,0,255])
            cv2.normalize(crop_hist,crop_hist,0,255,cv2.NORM_MINMAX)

            #设置对象的初始位置
            track_window=(boxes[0][0],boxes[0][1],boxes[1][0]-boxes[0][0],boxes[1][1]-boxes[0][1])
            cv2.imshow(windowNameSelection,crop)

        #重置盒子列表
        boxes=[]
    #选择区域进行展示，对别选择的对象做一个绿色的边框
    if selection_in_progress:
        top_left=(boxes[0][0],boxes[0][1])
        bottom_right=(current_mouse_position[0],current_mouse_position[1])
        cv2.rectangle(frame,top_left,bottom_right,(0,0,255),2)

    #如果已经选择区域
    if cropped:
        #转化整个输入图片为HSV形式
        img_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        #基于色调和饱和度的直方图反投影
        img_bproject=cv2.calcBackProject([img_hsv],[0,1],crop_hist,[0,180,0,255],1)
        cv2.imshow(windowName2,img_bproject)

        #利用camshift来预测新位置
        ret,track_window=cv2.CamShift(img_bproject,track_window,term_crit)

        #在图上绘制观测对象
        x,y,w,h=track_window
        frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        #提取观察对象的中心点
        pts=cv2.boxPoints(ret)
        pts=np.int0(pts)

        #使用卡尔曼滤波器
        kalman.correct(center(pts))
        #得到新的卡尔曼滤波器预测对象
        prediction=kalman.predict()
        #在图上进行绘制
        frame=cv2.rectangle(frame,(prediction[0]-(0.5*w),prediction[1]-(0.5*h)),(prediction[0]+(0.5*w),
                                        prediction[1]+(0.5*h)),(0,255,0),2)

    else:
        #在选择我们所使用的选择区域前
        img_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        mask=cv2.inRange(img_hsv,np.array((0.,float(s_lower),float(v_lower))),np.array((180.,
                                            float(s_upper),float(v_upper))))
        cv2.imshow(windowName2,mask)

    #展示图片
    cv2.imshow(windowName,frame)
    #停止加时器
    stop_timer=((cv2.getTickCount()-start_timer)/cv2.getTickFrequency())*1000

    # key=cv2.waitKey(max(2,40-int(math.ceil(stop_timer))))&0xFF
    key=cv2.waitKey(200)&0xFF

    if key==27:
        keep_processing=False
cv2.destroyAllWindows()
cap.release()


