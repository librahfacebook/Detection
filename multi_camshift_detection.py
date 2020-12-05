'''
使用Faster-RCNN检测行人并标记
再用Meanshift跟踪器进行多目标跟踪
同时利用卡尔曼滤波方法对位置进行预测
'''
import numpy as np
import cv2
import os
import collections
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# 系统环境设置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 指定要使用模型的名字(此处使用FasterRcnn)
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
# 指定模型路径
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# 数据集对应的label
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90


# 返回所选择矩形框的中心点
def center(points):
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4.0
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4.0
    return np.array([np.float32(x), np.float32(y)], np.float32)


# 定义展示的窗口名
windowName = "Kalman Object Tracking"
windowName2 = "Hue histogram back projection"

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

# 设置HSV选择阈值的滑块
s_lower = 60
s_upper = 255;
v_lower = 32;
v_upper = 255;

# 设置迭代次数，或者迭代10次或者至少移动1次
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


# 将训练好的模型以及标签加载到内存中，方便使用
def load():
    tf.reset_default_graph()
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    # 载入coco数据集标签文件,将其以index的方式读入内存中
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return category_index


# 定义session加载待测试的图片文件
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# 对原始图片进行目标检测定位的封装函数
def detect_objects(image_np, sess, detection_graph, category_index):
    # 定义结点，运行结果并可视化,扩充维度shape
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # boxes用来显示识别结果
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score代表识别出的物体与标签匹配的相似程度，在类型标签后面
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # 开始检测
    (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes,
                                                         num_detections], feed_dict={image_tensor: image_np_expanded})
    # 可视化结果
    _, new_boxes = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=10,
        min_score_thresh=.7,
        line_thickness=8)
    return image_np, new_boxes


# 对原始图片的处理
def process_image(image):
    category_index = load()
    detection_graph = tf.get_default_graph()
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_process = detect_objects(image, sess, detection_graph, category_index)
            return image_process


# 使用跟踪器对标记到的目标进行跟踪
def track_objects(video):
    # 初始化视频流
    cap = cv2.VideoCapture(video)

    # 创建窗口
    # cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    # cv2.namedWindow(windowName2, cv2.WINDOW_NORMAL)
    nums = 0  # 计算视频播放帧数
    times = 50  # 每隔多少ms播放帧
    # 循环帧并开始多目标跟踪
    flag = True
    while True:
        # 获取当前视频的帧
        ret, frame = cap.read()
        # 检查视频帧流是否结束
        if frame is None:
            break

        # 每隔60帧重新检测帧图像上的行人
        if nums % (3000 / times) == 0:
            flag = True
        if flag:
            # 初始化
            crop_hists = []
            track_windows = []
            # 绘制检测识别文字
            font = cv2.FONT_ITALIC
            cv2.putText(frame, 'Pedestrian Detection...', (50, 150), font, 1, (255, 255, 0), 2)
            cv2.imshow(windowName, frame)
            cv2.waitKey(1)
            # 检测该帧图像上的行人
            image, boxes = process_image(frame)
            vis_util.plt.imshow(image)
            vis_util.plt.show()
            # 处理检测到的box
            for box in boxes:
                box = tuple(box)
                x, y, w, h = box
                crop = frame[int(y):int(y + h), int(x):int(x + w)]
                h, w, c = crop.shape
                if h > 0 and w > 0:
                    # 将选择区域转换成HSV形式
                    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    # 选择所有色彩（0->180)
                    mask = cv2.inRange(hsv_crop, np.array((0., float(s_lower), float(v_lower))),
                                       np.array((180., float(s_upper), float(v_upper))))
                    # 构造一个色调和饱和度直方图并将其归一化
                    crop_hist = cv2.calcHist([hsv_crop], [0, 1], mask, [180, 255], [0, 180, 0, 255])
                    cv2.normalize(crop_hist, crop_hist, 0, 255, cv2.NORM_MINMAX)
                    crop_hists.append(crop_hist)
                    # 设置对象的初始位置
                    track_window = (int(x), int(y), int(w), int(h))
                    track_windows.append(track_window)
                    # 检查边界框并在帧上进行绘制
                    # cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
            flag = False

        # 使用CamShift进行目标跟踪并利用卡尔曼滤波器进行预测
        if not flag:
            for i in range(len(track_windows)):
                track_window = track_windows[i]
                crop_hist = crop_hists[i]
                # 转化整个输入图片为HSV形式
                img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # 基于色调和饱和度的直方图反投影
                img_bproject = cv2.calcBackProject([img_hsv], [0, 1], crop_hist, [0, 180, 0, 255], 1)
                cv2.imshow(windowName2, img_bproject)

                # 利用camshift来预测新位置
                ret, track_window = cv2.CamShift(img_bproject, track_window, term_crit)
                track_windows[i] = track_window

                # 在图上绘制观测对象
                x, y, w, h = track_window
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # 提取观察对象的中心点
                pts = cv2.boxPoints(ret)
                pts = np.int0(pts)

                # 使用卡尔曼滤波器
                kalman.correct(center(pts))
                # 得到新的卡尔曼滤波器预测对象
                prediction = kalman.predict()
                # 在图上进行绘制
                frame = cv2.rectangle(frame, (prediction[0] - (0.5 * w), prediction[1] - (0.5 * h)),
                                      (prediction[0] + (0.5 * w),
                                       prediction[1] + (0.5 * h)), (0, 255, 0), 2)

        # 显示框架以及选择要跟踪的对象
        cv2.imshow(windowName, frame)
        nums += 1

        key = cv2.waitKey(times) & 0xFF
        if key == ord("q"):
            break

    # 关闭所有窗口
    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    video = "test_videos/street.mp4"
    track_objects(video)
