import numpy as np
import os
import cv2
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# 系统环境设置
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#指定要使用模型的名字
MODEL_NAME = 'ssd_mobilenet_v1_coco_2018_01_28'
#指定模型路径
PATH_TO_CKPT=MODEL_NAME+'/frozen_inference_graph.pb'
#数据集对应的label
PATH_TO_LABELS=os.path.join('data','mscoco_label_map.pbtxt')
NUM_CLASSES=90

#将要测试的图片路径放入数组里
PATH_TO_TEST_IMAGES_DIR='test_images'
TEST_IMAGE_PATHS=[os.path.join(PATH_TO_TEST_IMAGES_DIR,'image{}.jpg'.format(i))
                  for i in range(2,4)]
#设置输出图片的大小
IMAGE_SIZE=(12,8)

#将训练好的模型以及标签加载到内存中，方便使用
def load():
    tf.reset_default_graph()
    od_graph_def=tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT,'rb') as fid:
        serialized_graph=fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def,name='')
    #载入coco数据集标签文件,将其以index的方式读入内存中
    label_map=label_map_util.load_labelmap(PATH_TO_LABELS)
    categories=label_map_util.convert_label_map_to_categories(label_map,max_num_classes=NUM_CLASSES,use_display_name=True)
    category_index=label_map_util.create_category_index(categories)

    return category_index

#定义session加载待测试的图片文件
def load_image_into_numpy_array(image):
    (im_width,im_height)=image.size
    return np.array(image.getdata()).reshape(
        (im_height,im_width,3)).astype(np.uint8)

#对原始图片进行目标检测定位的封装函数
def detect_objects(image_np,sess,detection_graph,category_index):
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
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    return image_np

#对原始图片的处理
def process_image(image):
    category_index = load()
    detection_graph = tf.get_default_graph()
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_process=detect_objects(image,sess,detection_graph,category_index)
            return image_process

#显示处理后的图片结果
def showImg():
    for image_path in TEST_IMAGE_PATHS:
        image=Image.open(image_path)
        print(image_path)
        plt.figure(figsize=IMAGE_SIZE)
        plt.subplot(1,2,1)
        plt.imshow(image)
        image_np=load_image_into_numpy_array(image)
        image_process=process_image(image_np)
        print(image_process.shape)
        plt.subplot(1,2,2)
        plt.imshow(image_process)
    plt.show()

if __name__ == '__main__':
    showImg()








