# from lerobot.eval.robot import RobotInferenceClient

import os
import sys 

sys.path.append("/home/binhng/Workspace/spcorl/lpb_libero")


from diffusion_policy.eval.robot import RobotInferenceClient
import datetime as dt
import numpy as np
client = RobotInferenceClient(port=6000)
client.reset()
while True:
    start_time = dt.datetime.now()
    # obs = {
    #     "observation.state": np.random.rand(1, 7).astype(np.float32),
    #     "observation.images.color.high": np.random.rand(1, 3, 480, 640).astype(np.float32),
    #     "observation.images.color.wrist_left": np.random.rand(1, 3, 480, 640).astype(np.float32),
    #     "observation.images.color.wrist_right": np.random.rand(1, 3, 480, 640).astype(np.float32),
    #     "task": ["test\n"],
    #     "inference_delay": 2,
    #     "prev_chunk_left_over": np.random.rand(1, 50, 7).astype(np.float32),
    #     "execution_horizon": 5
    # }
    
    # folding short
    # obs = {
    #     'high_images': np.random.randint(0, 255, (1, 2, 3, 480, 640), dtype=np.uint8),
    #     'wrist_left_images': np.random.randint(0, 255, (1, 2, 3, 480, 640), dtype=np.uint8),
    #     'wrist_right_images': np.random.randint(0, 255, (1, 2, 3, 480, 640), dtype=np.uint8),
    #     'states': np.zeros((1, 2, 14), dtype=np.float32)
    # }

    # placing drawer
    obs = {
        'high_images': np.random.randint(0, 255, (1, 2, 3, 480, 640), dtype=np.uint8),
        'wrist_images': np.random.randint(0, 255, (1, 2, 3, 480, 640), dtype=np.uint8),
        'states': np.zeros((1, 2, 13), dtype=np.float32)
    }
    # action = client.predict_action_chunk(obs)
    action = client.get_action(obs)
    print(action.keys())
    # print(action)
    print((dt.datetime.now() - start_time), action['action'].shape)