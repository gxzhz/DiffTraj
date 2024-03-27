import numpy as np

# 读取npy文件
data = np.load('../../dataset/head_50m.npy')

# 仅选择前六列的数据进行处理
selected_data = data[:, :6]

# 计算每列的均值和方差
column_means = np.mean(selected_data, axis=0)
column_std = np.std(selected_data, axis=0)

# 防止除以零，将标准差为0的列设为1，避免除法错误
column_std[column_std == 0] = 1

# 归一化每列数据
normalized_data = (selected_data - column_means) / column_std

# 计算每列的均值和方差
normalized_column_means = np.mean(normalized_data, axis=0)
normalized_column_std = np.std(normalized_data, axis=0)

# 将归一化后的数据与未归一化的后面几列数据拼接起来
final_data = np.concatenate((normalized_data, data[:, 6:]), axis=1)

# 保存归一化后的结果到npy文件
np.save('../../dataset/normalized/normalized_head_50m.npy', final_data)

# 保存每列的均值和方差到npy文件
np.save('../../dataset/normalized/head_means_50m.npy', column_means)
np.save('../../dataset/normalized/head_std_50m.npy', column_std)

# 打印均值和方差到标准输出设备
print("前六列的均值：", column_means)
print("前六列的方差：", column_std)
