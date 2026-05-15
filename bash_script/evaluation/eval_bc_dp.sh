#!/bin/bash

GPU=1

export MUJOCO_GL="glx"

OUTDIR="eval_logs"

NOISE_SCHEDULER="ddpm"
NUM_INFERENCE_STEPS=100

ALL_DATASETS=(
  "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"
  "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5"
  "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5"
  "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5"
  "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5"
  "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5"
  "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5"
  "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5"
  "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5"
  "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy_demo.hdf5"
)


POLICY_CKPT="logs/reproduce/libero_image/dp_cnn_ERate.1/dp_cnn_ERate.1_42/checkpoints/50.ckpt"

for DATASET in "${ALL_DATASETS[@]}"; do
  TASK=${DATASET%.hdf5}
  for SEED in 1; do
    CUDA_VISIBLE_DEVICES=${GPU} python eval_libero.py \
      --checkpoint ${POLICY_CKPT} \
      --output_dir "${OUTDIR}/${TASK}" \
      --noise_scheduler ${NOISE_SCHEDULER} --num_inference_steps ${NUM_INFERENCE_STEPS} \
      --dataset_name ${DATASET} \
      --ntest 50 \
      --seed ${SEED}
  done
done
