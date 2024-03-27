import numpy as np

# # 读取三维npy文件
# data_3d = np.load('../dataset/traj_linear_200.npy')
#
# # 存储归一化后的数据、均值和方差
# normalized_data = []
# column_means = []
# column_std = []
#
# # 对每个二维数组进行处理
# for i in range(data_3d.shape[0]):
#     # 获取当前二维数组
#     data_2d = data_3d[i]
#
#     # 计算每列的均值和方差
#     means = np.mean(data_2d, axis=0)
#     stds = np.std(data_2d, axis=0)
#
#     # 防止除以零，将标准差为0的列设为1，避免除法错误
#     stds[stds == 0] = 1
#
#     # 归一化每列数据
#     normalized_data_2d = (data_2d - means) / stds
#
#     # 计算每列的均值和方差
#     normalized_means = np.mean(normalized_data_2d, axis=0)
#     normalized_stds = np.std(normalized_data_2d, axis=0)
#
#     # 存储归一化后的数据、均值和方差
#     normalized_data.append(normalized_data_2d)
#     column_means.append(means)
#     column_std.append(stds)
#
# # 转换为数组
# normalized_data = np.array(normalized_data)
# column_means = np.array(column_means)
# column_std = np.array(column_std)
#
# # 保存归一化后的结果到npy文件
# np.save('../dataset/normalized/normalized_traj_200.npy', normalized_data)
#
# # 保存每列的均值和方差到npy文件
# np.save('../dataset/normalized/traj_means_200.npy', column_means)
# np.save('../dataset/normalized/traj_std_200.npy', column_std)


# 读取三维npy文件
data_3d = np.load('../../dataset/traj_linear_10.npy')

# 将三维数据reshape为二维
data_2d = data_3d.reshape(-1, 2)

# 计算每列的均值和方差
column_means = np.mean(data_2d, axis=0)
column_std = np.std(data_2d, axis=0)

# 防止除以零，将标准差为0的列设为1，避免除法错误
column_std[column_std == 0] = 1

# 归一化每列数据
normalized_data_2d = (data_2d - column_means) / column_std

# 计算每列的均值和方差
normalized_means = np.mean(normalized_data_2d, axis=0)
normalized_stds = np.std(normalized_data_2d, axis=0)

# 将二维数据reshape回原来的三维形状
normalized_data_3d = normalized_data_2d.reshape(data_3d.shape)


# 保存归一化后的结果到npy文件
np.save('../../dataset/normalized/normalized_traj_10.npy', normalized_data_3d)

# 保存每列的均值和方差到npy文件
np.save('../../dataset/normalized/traj_means_10.npy', column_means)
np.save('../../dataset/normalized/traj_std_10.npy', column_std)

# 打印均值和方差到标准输出设备
print("轨迹均值：", column_means)
print("轨迹方差：", column_std)
