'''
基于自适应混合高斯背景建模的背景减除法,将前景图的轮廓进行绘制并展示
'''
import matplotlib.pyplot as plt
import cv2
import numpy as np

#打开视频
file = "test_videos/singleball.mov"
cap=cv2.VideoCapture(file)
# ret=True
# while ret:
#     ret,frame=cap.read()
#     if ret==True:
#         cv2.imshow("Original Video",frame)
#         cv2.waitKey(100)
#     else:
#         break
# cv2.destroyAllWindows()
numFrames=cap.get(7)
print("Number of Frames in the video:",numFrames)

#将每个帧检测到的对象位置绘制到图中
plt.figure()
# plt.hold(True)
plt.axis([0,cap.get(3),cap.get(4),0])
count=0 #计算帧数
bgs=cv2.createBackgroundSubtractorMOG2()#基于自适应混合高斯背景建模的背景减除法
measuredTrack=np.zeros((int(numFrames),2))-1#初始化每个帧的小球坐标位置
while count<numFrames:
    count+=1
    ret,img2=cap.read()
    cv2.imshow("Video",img2)
    cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
    foremat=bgs.apply(img2,learningRate=0.01) #返回当前帧的前景蒙版
    cv2.waitKey(20)
    ret,thresh=cv2.threshold(foremat,220,255,0)#通过将阈值应用于前景蒙版，将其转换为二进制图像，前景处为1，后景为0
    im2,contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#确定前景对象的轮廓

    #假设轮廓的质心是前景对象位置，跟踪每个帧的位置，对于每个帧，被跟踪对象位置存储在numpy数组中
    if len(contours)>0:
        for i in range(len(contours)):
            area=cv2.contourArea(contours[i])#计算轮廓面积
            if area>100:
                m=np.mean(contours[i],axis=0)#寻找物体的中心点
                measuredTrack[count-1,:]=m[0]
                plt.plot(m[0,0],m[0,1],'xr')
    cv2.imshow('Foreground',foremat)
    cv2.namedWindow('Foreground',cv2.WINDOW_NORMAL)
    cv2.waitKey(160)
cap.release()
cv2.destroyAllWindows()
print(measuredTrack)
#保存小球滚动 的足迹
np.save('ballTrajectory',measuredTrack)
plt.axis((0,480,360,0))
plt.show()

