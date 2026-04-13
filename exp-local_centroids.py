import os
from tqdm import tqdm
from utils import create_batch, get_model, plot_img, normalize_img, process_gradient
from centroids import Centroids, LocalCentroids, InternalModuleWrapper

DEVICE='cuda'

# dataset parameters
DATA_DIR='./data'
DATASET='imagenet'
SEED=5
BATCH_SIZE=2


# local centroid parameters
STDEV_SPREAD=0.15
N_SAMPLES=30

# model parameters
ARCH = 'convnext_large'
RB_MODEL = 'RodriguezMunoz2024Characterizing_Swin-B'
TARGET_MODULE = 'model.layers.1.blocks.1.drop_path2'
OUTPUT_DIR = f'outputs/{RB_MODEL}/{DATASET}' if len(RB_MODEL)>0 else f'outputs/{ARCH}/{DATASET}'
os.makedirs(OUTPUT_DIR,exist_ok=True)

if len(RB_MODEL)>0:
    MU = None
    SIGMA = None
else:
    MU = (0.485, 0.456, 0.406)
    SIGMA = (0.229, 0.224, 0.225)

xs, indices = create_batch(batch_size=BATCH_SIZE,dataset=DATASET,data_dir=DATA_DIR,seed=SEED)
xs=xs.to(DEVICE)

model = get_model(ARCH,RB_MODEL)
model.to(DEVICE)

for x,i in zip(xs,indices):
    ximg=x.permute(1,2,0).cpu()
    plot_img(ximg,f'{OUTPUT_DIR}/inp_{i}.png')

# centroids of pretrained model

analyzer=Centroids(model)

for x, i in tqdm(zip(xs,indices),total=len(xs)):
    x=normalize_img(x, MU, SIGMA)
    c=analyzer(x)[0]
    
    cimg=process_gradient(c)

    plot_img(cimg,f'{OUTPUT_DIR}/pretrained_centroid_{i}.png',cmap='grey')

# local centroids of pretrained model

analyzer=LocalCentroids(model, stdev_spread=STDEV_SPREAD, n_samples=N_SAMPLES)

for x, i in tqdm(zip(xs,indices),total=len(xs)):
    
    x=normalize_img(x, MU, SIGMA)
    c=analyzer(x)[0]
    
    cimg=process_gradient(c)

    plot_img(cimg,f'{OUTPUT_DIR}/pretrained_localcentroid_{i}.png',cmap='grey')

# local centroid of sub-component

sub_model=InternalModuleWrapper(model,TARGET_MODULE)

analyzer=LocalCentroids(sub_model, stdev_spread=STDEV_SPREAD, n_samples=N_SAMPLES)

for x, i in tqdm(zip(xs,indices),total=len(xs)):
    x=normalize_img(x, MU, SIGMA)
    c=analyzer(x)[0]
    
    cimg=process_gradient(c)

    plot_img(cimg,f'{OUTPUT_DIR}/pretrained_localcentroid-{TARGET_MODULE}_{i}.png',cmap='grey')

# local centroids of randomly initialized model

model = get_model(ARCH,RB_MODEL,pretrained=False)
model.to(DEVICE)
analyzer=LocalCentroids(model, stdev_spread=STDEV_SPREAD, n_samples=N_SAMPLES)

for x, i in tqdm(zip(xs,indices),total=len(xs)):
    x=normalize_img(x, MU, SIGMA)
    c=analyzer(x)[0]
    
    cimg=process_gradient(c)

    plot_img(cimg,f'{OUTPUT_DIR}/random_localcentroid_{i}.png',cmap='grey')