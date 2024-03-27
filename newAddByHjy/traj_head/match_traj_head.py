import pandas as pd
import numpy as np

# 读取 head 和 traj_head 文件
head_df = pd.read_csv('../../dataset/reNumberHead_50m.csv')
traj_df = pd.read_csv('../../dataset/traj_linear_10.csv')

'匹配'
# 将 head 数据附加到 traj_head 数据的末尾，根据 id 列进行匹配
merged_df = pd.merge(traj_df, head_df, on='id')
merged_df.to_csv('../dataset/match_50m.csv', index=False)

'分离'
# 按序排列的traj和head
new_traj_df = merged_df.iloc[:, :3]  # .to_csv('../datasetInit/new_traj.csv', index=False)
# 按照 'id' 列进行分组，并取每组的最后八列作为头部数据，同时保留第一列
new_head_df = merged_df.groupby('id').last().reset_index().iloc[:, :1].join(
    merged_df.groupby('id').last().reset_index().iloc[:, -8:])

'处理traj'
# traj进行分组
traj_groups = new_traj_df.groupby('id')

# 初始化一个空列表来存储分组后的数据
traj_grouped_data = []

# 遍历每个分组，并将经纬度数据保存到 NumPy 数组中
for _, group_df in traj_groups:
    lon_lat_array = group_df[['lon', 'lat']].values
    traj_grouped_data.append(lon_lat_array)

# 将列表转换为 NumPy 数组，得到一个三维数组
grouped_data_array = np.array(traj_grouped_data)

# 如果需要，可以将 NumPy 数组保存到本地
np.save('../../dataset/traj_linear_10.npy', grouped_data_array)

'处理head'
head_match = new_head_df.iloc[:, 1:]  # 选择除第一列之外的所有列
head_match = head_match.astype(float)  # 将数据转换为浮点数类型
np.save('../../dataset/head_50m.npy', head_match.to_numpy())  # 保存NumPy数组到文件
