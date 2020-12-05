# Detection
## 基于视频的行人流量密度检测

1.detection.py:通过已经训练好的Faster-Rcnn参数实现对行人的识别并标记（其中标记行人的阈值为0.7，即识别率必须达到70%）；<br>
2.camshift2.py:利用mean-shift对已经标记的人进行目标跟踪，中间通过不断迭代更新行人目标位置并实时标记；<br>
3.kalmon.py:借助卡尔曼滤波方式来对行人移动位置进行预测，提高目标跟踪的精度；<br>
4.multi-object-tracking.py:利用多对象目标跟踪器实现对多个目标进行跟踪。<br>
5. multi_camshift_detection.py:使用camshift方法进行目标跟踪，并利用卡尔曼滤波方法进行目标预测，从而实现多目标跟踪。（此方法目标标记框移动幅度较大）
<br>6. multi_track_detection.py：正式完成的基于视频的行人目标跟踪及流量密度检测程序。

## 附属依赖库：
faster_rcnn_inception_v2_coco_2018_01_28（faster-rcnn网络框架）、
ssd_mobilenet_v1_coco_2018_01_28（ssd-mobilenet网络框架）、object_detection（目标检测库，为符合本程序使用中间有参数修改）

## 说明：本项目为基于视频的行人流量密度检测，所采用的编程语言为python，版本为3.6.4，所使用的主要工具库为opencv3.4。



## 主要使用方法：
(1) 图像预处理常用算法研究； 
(2) 背景建模算法研究； 
(3) 运动目标检测算法研究； 
(4) 目标匹配与跟踪算法

## 更新
针对标记区域内的行人检测数量统计，更新为判断某点是否在四边形区域内部，主要代码见multi_track_detection.py的函数isPosition(center_position)。





## 结果展示：
![](https://github.com/librahfacebook/Detection/blob/master/result_images/result.jpg) 

## 数据分析图：<br>
![](https://github.com/librahfacebook/Detection/blob/master/result_images/analysis.png)
