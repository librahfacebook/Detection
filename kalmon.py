'''
Kalmon算法：在用于对象跟踪的卡尔曼滤波算法的帮助下载场景中滚动
'''
from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt

# 加载保存好的小球移动文件
Measured = np.load("ballTrajectory.npy")
print(Measured.shape)
# print(Measured)

# 取出视频当中刚开始无小球的部分
while True:
    if Measured[0, 0] == -1.:
        Measured = np.delete(Measured, 0, 0)
    else:
        break
numMeas = Measured.shape[0]
print(Measured.shape)
# print(Measured)

# 使用卡尔曼滤波器来预测小球中间被阻挡住的位置
MarkedMeasure = np.ma.masked_less(Measured, 0)  # 屏蔽掉无坐标部分
# print(MarkedMeasure)

# 卡尔曼滤波器测量参数
Transition_Matrix = [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]  # 转移矩阵
Observation_Matrix = [[1, 0, 0, 0], [0, 1, 0, 0]]  # 观察矩阵
# 其他参数
# 初始状态
xinit = MarkedMeasure[0, 0]  # 当前位置的x坐标
yinit = MarkedMeasure[0, 1]  # 当前位置的y坐标
vxinit = MarkedMeasure[1, 0] - MarkedMeasure[0, 0]  # x方向的当前速度
vyinit = MarkedMeasure[1, 1] - MarkedMeasure[0, 1]  # y方向的当前速度
initstate = [xinit, yinit, vxinit, vyinit]
initcovariance = 1.0e-3 * np.eye(4)  # 初始状态协方差，描述了初始状态的确定性
transistionCov = 1.0e-4 * np.eye(4)  # 过渡协方差，描述了过程模型的确定性
observationCov = 1.0e-1 * np.eye(2)  # 观测协方差，描述了观测模型的确定性

kf = KalmanFilter(transition_matrices=Transition_Matrix,
                  observation_matrices=Observation_Matrix,
                  initial_state_mean=initstate,
                  initial_state_covariance=initcovariance,
                  transition_covariance=transistionCov,
                  observation_covariance=observationCov)
# 通过调用KalmanFilter的filter()方法，计算轨道以及filtered_state_covariances的正确性
(filtered_state_means, filtered_state_covariance) = kf.filter(MarkedMeasure)
plt.plot(MarkedMeasure[:, 0], MarkedMeasure[:, 1], 'xr', label='measured')
plt.axis([0, 520, 360, 0])
# plt.hold(True)
plt.plot(filtered_state_means[:, 0], filtered_state_means[:, 1], 'ob', label='kalman output')
# plt.hold(True)
plt.legend(loc=3)
plt.title('Constant Velocity Kalman Filter')
plt.show()
