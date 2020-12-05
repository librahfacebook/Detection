'''
先使用Faster-RCNN检测行人并标记
再用多目标跟踪器进行跟踪
'''
import numpy as np
import cv2
import os
import tensorflow as tf
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 系统环境设置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 指定要使用模型的名字(此处使用FasterRcnn)
MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
# 指定模型路径
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# 数据集对应的label
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
# 检测视频窗口
WINDOW_NAME = 'Pedestrian'
# 反向投影视频窗口
WINDOW_NAME2 = "Hue histogram back projection"
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
# 行人检测选择区域
BORDER = [[142, 171], [101, 339], [283, 339], [296, 171]]


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
        max_boxes_to_draw=20,
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


# 返回所选择矩形框的中心点
def center(box):
    (x, y, w, h) = box
    center_x = int(x + w / 2.0)
    center_y = int(y + h / 2.0)
    return (center_x, center_y)

# 求取向量叉乘
def get_vector_cross_product(position0, position1, position):

    product_value = (position1[0]-position0[0]) * (position[1]-position0[1]) -       (position1[1]-position0[1])*(position[0]-position0[0])

    return product_value

# 判断该点是否在四边形内部
def isPosition(center_position):

    directions = []
    isPosition = True
    for i in range(0, len(BORDER)):
        direction = get_vector_cross_product(BORDER[i], BORDER[(i+1)%len(BORDER)], center_position)
        directions.append(direction)

    for i in range(0, len(directions)-1):
        if directions[i]*directions[i+1] < 0:
            isPosition = False
            break
    
    return isPosition

# 绘制直方图和折线图（每次检测到所经过选择区域的行人数）
def histograms_line(peoples):
    plt.subplots_adjust(hspace=0.45)
    plt.subplot(2, 1, 1)
    x = [i for i in range(1, len(peoples) + 1)]
    plt.bar(x, peoples)
    plt.xlabel("检测区间数")
    plt.ylabel("行人数")
    plt.title("行人检测数分布柱状图")
    plt.subplot(2, 1, 2)
    plt.scatter(x, peoples, s=100)  # 散点图
    plt.plot(x, peoples, linewidth=2)
    plt.xlabel("检测区间数")
    plt.ylabel("行人数")
    plt.title("行人检测数分布折线图")
    plt.show()


# 使用跟踪器对标记到的目标进行跟踪
def track_objects(video, object_tracker):
    # 初始化视频流
    cap = cv2.VideoCapture(video)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # 帧速率
    print("视频帧速率：", frame_rate)
    frame_counts = cap.get(cv2.CAP_PROP_FRAME_COUNT)  # 总帧长
    print("视频总帧长：", frame_counts)
    video_time = frame_counts / frame_rate  # 视频总时间
    print("视频总时间：{}s".format(video_time))
    nums = 0  # 计算视频播放帧数
    times = 40  # 每隔多少ms播放帧
    detection_nums = -1  # 检测次数
    peoples = []  # 经过选择区域的行人数
    # 在视频帧上绘制分界线进行计数
    pts = np.array(BORDER, np.int32)
    pts = pts.reshape((-1, 1, 2))
    # 循环帧并开始多目标跟踪
    flag = True
    while True:
        # 获取当前视频的帧
        ret, frame = cap.read()
        # 检查视频帧流是否结束
        if frame is None:
            break
        # 将当前帧重置 (加快处理速度)
        # frame = imutils.resize(frame, width=600)
        # 每隔100帧重新检测帧图像上的行人
        if nums % (4000 / times) == 0:
            peoples.append(0)
            detection_nums += 1
            flag = True
        # 对于每一个被跟踪的对象矩形框进行更新
        if not flag:
            (success, boxes) = trackers.update(frame)
        if flag:
            # 重新初始化多对象跟踪器
            trackers = cv2.MultiTracker_create()
            # 绘制检测识别文字
            font = cv2.FONT_ITALIC
            h, w, c = frame.shape
            cv2.putText(frame, 'Pedestrian Detection...', (int(w * 0.4), int(h * 0.85)), font, 1, (255, 255, 0), 2)
            cv2.polylines(frame, [pts], True, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, frame)
            cv2.waitKey(1)
            # 检测该帧图像固定区域上的行人
            # selected_frame=frame[154:426,85:300]
            _, boxes = process_image(frame)

            # 将所有标记对象先加入到多对象跟踪器中
            for box in boxes:
                box = tuple(box)
                # 创建一个新的对象跟踪器为新的边界框并将它添加到多对象跟踪器里
                tracker = OPENCV_OBJECT_TRACKERS[object_tracker]()
                trackers.add(tracker, frame, box)
            # 展示最开始的帧检测图
            # vis_util.plt.imshow(frame)
            # vis_util.plt.show()
            flag = False

        cv2.polylines(frame, [pts], True, (255, 0, 0), 1, cv2.LINE_AA)
        # 检查边界框并在帧上进行绘制
        for box in boxes:
            box = tuple(box)
            (x, y, w, h) = box
            center_position = center(box)
            # 绘制矩形框
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            # 绘制矩形框的质心
            cv2.circle(frame, center(box), 2, (0, 0, 255), 2)
            # 计算进出选择区域的人数
            if isPosition(center_position):
                peoples[detection_nums] += 1

        # 显示框架以及选择要跟踪的对象
        cv2.imshow(WINDOW_NAME, frame)
        nums += 1
        key = cv2.waitKey(times) & 0xFF
        if key == ord("q"):
            break

    # 关闭所有窗口
    cv2.destroyAllWindows()
    cap.release()
    return peoples, video_time


if __name__ == '__main__':
    video = "./test_videos/street.mp4"
    object_tracker = "kcf"
    peoples, times = track_objects(video, object_tracker)
    print(peoples)
    # peoples = [1, 1, 2, 4, 5, 6, 2, 4, 1]
    # times = 70
    total_peoples = 0
    for people in peoples:
        total_peoples += people
    print("总人数：", total_peoples)
    per_peoples = total_peoples / (times / 60.0)
    print("行人密度（每分钟走过的人数）：", per_peoples)
    # 直方图折线图显示
    histograms_line(peoples)
#                        _oo0oo_
#                       o8888888o
#                       88" . "88
#                       (| -_- |)
#                       0\  =  /0
#                     ___/`---'\___
#                   .' \\|     |// '.
#                  / \\|||  :  |||// \
#                 / _||||| -:- |||||- \
#                |   | \\\  -  /// |   |
#                | \_|  ''\---/''  |_/ |
#                \  .-\__  '-'  ___/-. /
#              ___'. .'  /--.--\  `. .'___
#           ."" '<  `.___\_<|>_/___.' >' "".
#          | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#          \  \ `_.   \_ __\ /__ _/   .-` /  /
#      =====`-.____`.___ \_____/___.-`___.-'=====
#      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
