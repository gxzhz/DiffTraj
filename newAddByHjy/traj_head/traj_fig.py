import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

trajs = np.load('../../dataset/traj.npy',
                allow_pickle=True)

trajs = pd.DataFrame(trajs)

trajs_group = trajs.groupby(0)

plt.figure(figsize=(8,8))
for group_id, group_df in trajs_group:
    plt.plot(group_df[1], group_df[2],color='blue',alpha=0.05)
plt.tight_layout()
plt.title('Wuhan_traj')
plt.savefig('Wuhan_traj.png')
plt.show()


trajs_linear = np.load('../../dataset/traj_linear_200.npy',
                       allow_pickle=True)
trajs_linear = trajs_linear[:, :, :2]

plt.figure(figsize=(8,8))
for i in range(len(trajs_linear)):
    traj=trajs_linear[i]
    plt.plot(traj[:,0],traj[:,1],color='blue',alpha=0.05)
plt.tight_layout()
plt.title('Wuhan_traj_linear')
plt.savefig('Wuhan_traj_linear_200.png')
plt.show()