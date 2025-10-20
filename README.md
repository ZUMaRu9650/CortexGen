# CortexGen

This repository contains the source code for the MICCAI 2025 paper [CortexGen: A Geometric Generative Framework for Realistic Cortical Surface Generation Using Latent Flow Matching](https://papers.miccai.org/miccai-2025/paper/1498_paper.pdf)
<img width="960" alt="幻灯片2" src="https://github.com/user-attachments/assets/fb919d38-81fd-4309-b188-72d40a72e4ea" />


## Outline

* **source**
  
  ```VAE.py``` and ```layers.py``` contains implementation of **GVAE**
  
  ```FlowMatching.py``` contains implementation of **Latent Flow Matching**

  ```step.py``` contains the functions for trainging **CortexGen** from scratch

  ```loss.py``` contains implementation of **mesh Laplacian loss** and **normal consistency loss** 

  ```dataset.py``` contains a subclass of ```torch.utils.data.Dataset```

  ```utils.py``` mainly contains some useful functions for data processing. Note, please change the data loading path (these data are located in ```necessary_data/neigh_indices``` and ```necessary_data/ico```) in these functions, including ```Get_neighs_order```, ```Get_upconv_index```, ```Get_new_edges```.

* **pretrained_model** contains the pretrained model parameters
* **necessary_data** contains the data needed for model calculations
* ```train_gvae_L.py```, ```train_gvae_H.py```, ```train_fm_L.py```, ```train_fm_H.py```, ```train_fm_H_ft.py``` contains scripts that demonstrate how to train _GVAE<sub>L</sub>_, _GVAE<sub>H</sub>_, _v<sub>θL</sub>_, _v<sub>θH</sub>_, and how to finetune _v<sub>θH</sub>_.

## Tips

Each input cortical surface needs to be preprocessed through corresponding functions (i.e., ```read_vtk``` and ```adjacency```) in ```utils.py``` like:
```
### Assuming you already have access to an icosahedron-reparameterized original cortical surface (saved in ```.vtk``` format), which shares the same number of vertices and connectivity as a standard ico7 sphere ###

vtk_ico7 = read_vtk("/your_data/ico7_reparameterized_cortical_surface.vtk")
ico7_vertices = vtk_ico7['vertices']  # [163842, 3]
ico5_vertices = ico7_vertices[0:10242, :]  # [10242, 3]
f = np.load("your_data/CortexGen/necessary_data/ico/faces_ico5.npy")
adj_matrix, adj_degree = adjacency(torch.from_numpy(f))
lap_ico5_vertices = torch.bmm(adj_matrix, torch.from_numpy(ico5_vertices).float().unsqueeze(0)) / adj_degree

d = dict()
d["ico7"] = ico7_vertices
d["ico5"] = lap_ico5_vertices

np.save("your_data/ico7_cortical_surface.dict", d)
```
We suggest you perform the above processing for each input mesh in advance and save the results in ```.dict.npy``` format.

## Arbitrarily generate cortical surfaces using pre-trained model parameters
```
import os
import numpy as np
import torch
import time
from tqdm import tqdm
from source.VAE import GVAE
from source.FlowMatching import FM
from source.utils import write_vtk, coarse2fine, Get_new_edges

num_surfaces = 100
device = 'cuda:0'
f_ico5 = np.load("/your_work_space/CortexGen/necessary_data/ico/faces_ico5.npy")
f_ico7 = np.load("/your_work_space/CortexGen/necessary_data/ico/faces_ico7.npy")
ico7 = np.load("/your_work_space/CortexGen/necessary_data/ico/ico7.npy")
ico5 = ico7[0:10242] / 100
ico5 = torch.from_numpy(ico5).to(device)
new_edges = Get_new_edges()

######## coarse ########

gvae_L = GVAE(in_ch=3, start_idx=2, num_resolution=3, base_num_ch=64, latent_dim=8, num_hid=128, time=0.1, tol=1e-5)
gvae_L.eval()
gvae_L.to(torch.device(device))
gvae_L.load_state_dict(torch.load("/your_work_space/CortexGen/pretrained_model/gvae_L.pkl", map_location=device))

fm_L = FM(start_res_idx=4, channels=128, channel_mul=[1, 2, 2, 4], attn=[False, False, False, True], latent_dim=8, condition_dim=0, device=device)
fm_L.eval()
fm_L.to(torch.device(device))
fm_L.load_state_dict(torch.load("/your_work_space/CortexGen/pretrained_model/fm_L.pkl", map_location=device))

######## fine ########

gvae_H = GVAE(in_ch=3, start_idx=0, num_resolution=3, base_num_ch=64, latent_dim=8, num_hid=128, time=0.2, tol=1e-5)
gvae_H.eval()
gvae_H.to(torch.device(device))
gvae_H.load_state_dict(torch.load("/your_work_space/CortexGen/pretrained_model/gvae_H.pkl", map_location=device))

fm_H = FM(start_res_idx=2, channels=128, channel_mul=[1, 2, 2, 4], attn=[False, False, False, True], latent_dim=8, condition_dim=3, device=device)
fm_H.eval()
fm_H.to(torch.device(device))
fm_H.load_state_dict(torch.load("/your_work_space/CortexGen/pretrained_model/fm_H_ft.pkl", map_location=device))

t1 = time.time()
with torch.no_grad():
    for i in tqdm(range(num_surfaces)):
        ######## coarse ########
        z_L = fm_L.sample(1, 642, 8, 10, None)
        sample_L = gvae_L.decode_z(z_L, ico5)
        
        
        ######## fine ########
        up_sample_L = coarse2fine(sample_L, 2, new_edges)
        z_H = fm_H.sample(1, 10242, 8, 1, sample_L)
        sample_H = gvae_H.decode_z(z_H, up_sample_L)
        
        d = dict()
        d['vertices'] = sample_H[0].cpu().numpy()
        d['faces'] = np.concatenate((3*np.ones(f_ico7.shape[0],dtype=int).reshape(-1,1), f_ico7), axis=1)
           
        write_vtk(d, "vertices", os.path.join("/your_save_data_space", "sample_" + str(i+1) + ".vtk"))
t2 = time.time()
print(t2 - t1)
```

Note, a lower proportion of self-intersecting faces generally indicates better visual quality of the generated cortical surface.
