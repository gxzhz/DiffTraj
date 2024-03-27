args = {
    'data': {
        'dataset': 'Wuhan',
        'traj_path1': '../dataset/normalized/normalized_traj_10.npy',
        'head_path2': '../dataset/normalized/normalized_head_50m.npy',
        'traj_length': 10,
        'channels': 2,
        'uniform_dequantization': False,
        'gaussian_dequantization': False,
        'num_workers': False,
    },
    'model': {
        'type': "simple",
        'attr_dim': 8,
        'guidance_scale': 3,
        'in_channels': 2,
        'out_ch': 2,
        'ch': 128,
        'ch_mult': [1, 2, 2, 2],
        'num_res_blocks': 2,
        'attn_resolutions': [16],
        'dropout': 0.1,
        'var_type': 'fixedlarge',
        'ema_rate': 0.9999,
        'ema': True,
        'resamp_with_conv': True,
    },
    'diffusion': {
        'beta_schedule': 'linear',
        'beta_start': 0.0001,
        'beta_end': 0.05,
        'num_diffusion_timesteps': 500,
    },
    'training': {
        'batch_size': 1024,
        'n_epochs': 10,
        'n_iters': 5000000,
        'snapshot_freq': 5000,
        'validation_freq': 2000,
    },
    'sampling': {
        'batch_size': 64,
        'last_only': True,
    }
}