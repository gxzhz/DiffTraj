import pandas as pd
import numpy as np

# 先用pandas读入csv
data = pd.read_csv("../../dataset/traj.csv")
# 再使用numpy保存为npy
np.save("../../dataset/traj.npy", data)
