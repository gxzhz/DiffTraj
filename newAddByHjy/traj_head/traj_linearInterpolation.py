import pandas as pd
import numpy as np

from utils.utils import resample_trajectory

traj_df = pd.read_csv('../../dataset/traj.csv')

# 按照 'id' 列进行分组
grouped = traj_df.groupby('id')

# 初始化一个空列表，用于存储经过处理的组
processed_groups = []

# 遍历每个组，进行处理
for group_id, group_df in grouped:
    # 获取第一列和第二三列数据
    traj_columns = group_df.iloc[:, 1:3].values
    # 对每组进行线性插值
    traj_groups = resample_trajectory(traj_columns, 10).astype(str)
    new_id_column = np.full((1, 1), group_id)
    # 在 resampled_traj 的第一列插入 ID
    processed_groups.append(pd.DataFrame(np.insert(traj_groups, 0, new_id_column, axis=1)))


# 将经过处理的数据并成一个新的 DataFrame
processed_df = pd.concat(processed_groups)

# 修改所有列名，使用DataFrame的columns属性来指定新的列名列表
processed_df.columns = ['id', 'lon', 'lat']

processed_df.to_csv('../dataset/traj_linear_10.csv', index=False)
