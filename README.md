

## Install environment


```bash

git clone https://github.com/tunglm2203/lpb_libero
git checkout collect_data

conda env create -f conda_environment.yaml
uv pip install transformer==4.30.2 huggingface_hub==0.20.3 bddl easydict
conda install -c conda-forge mesalib glfw glew patchelf
```

## Download expert and rollout dataset
expert: https://huggingface.co/datasets/ducido/LIVING_ROOM_SCENE6
rollout: https://huggingface.co/datasets/ducido/LIVING_ROOM_SCENE6

```bash

# Expert
git clone https://huggingface.co/datasets/ducido/LIVING_ROOM_SCENE6 data/libero_10/libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo

# Rollout
git clone https://huggingface.co/datasets/ducido/LIVING_ROOM_SCENE6_rollout_200eps /logs/collect_data_200eps/libero_10/datacollect_diffusion_unet_libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo
```


### training

```
bash bash_script/training/train_pbrl.sh
```



