import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from utils.Traj_UNet import *
from utils.config import args
from utils.utils import *

temp = {}
for k, v in args.items():
    temp[k] = SimpleNamespace(**v)

config = SimpleNamespace(**temp)

unet = Guide_UNet(config)
# load the model
unet.load_state_dict(torch.load(
    '../DiffTraj/Wuhan_steps=500_len=10_0.05_bs=1024/models/03-20-14-32-54/unet_10_normalized.pt'))

n_steps = config.diffusion.num_diffusion_timesteps
beta = torch.linspace(config.diffusion.beta_start,
                          config.diffusion.beta_end, n_steps)
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha, dim=0)
lr = 2e-4  # Explore this - might want it lower when training on the full dataset

eta=0.0
timesteps=100
skip = n_steps // timesteps
seq = range(0, n_steps, skip)

# load head information for guide trajectory generation
batchsize = 500
head = np.load('../dataset/head_50m.npy',
                   allow_pickle=True)
head = torch.from_numpy(head).float()
dataloader = DataLoader(head, batch_size=batchsize, shuffle=True, num_workers=0)

Gen_traj = []
Gen_head = []
for i in tqdm(range(1)):
    head = next(iter(dataloader))
    lengths = head[:, 3]
    lengths = lengths.int()
    tes = head[:,:6].numpy()
    Gen_head.extend((tes))
    # Start with random noise
    x = torch.randn(batchsize, 2, config.data.traj_length)
    ims = []
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        with torch.no_grad():
            pred_noise = unet(x, t, head)
            # print(pred_noise.shape)
            x = p_xt(x, pred_noise, t, next_t, beta, eta)
            if i % 10 == 0:
                ims.append(x.cpu().squeeze(0))
    trajs = ims[-1].cpu().numpy()
    trajs = trajs[:,:2,:]
    # resample the trajectory length
    for j in range(batchsize):
        new_traj = resample_trajectory(trajs[j].T, lengths[j])
        new_traj = new_traj
        Gen_traj.append(new_traj)
    break


plt.figure(figsize=(8,8))
for i in range(len(Gen_traj)):
    traj=Gen_traj[i]
    plt.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
plt.tight_layout()
plt.savefig('gen_Wuhan_traj.png')
plt.show()