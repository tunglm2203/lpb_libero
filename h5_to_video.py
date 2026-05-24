import os
import h5py
import imageio
import numpy as np
from tqdm import tqdm


def save_demo_videos(
    file_path,
    output_dir="output_videos",
    image_key="agentview_rgb",
    fps=20,
):
    """
    Save each demo in LIBERO hdf5 file as a separate mp4 video.

    Args:
        file_path (str): path to hdf5 file
        output_dir (str): directory to save videos
        image_key (str): observation image key
        fps (int): output video fps
    """

    os.makedirs(output_dir, exist_ok=True)

    f = h5py.File(file_path, "r")

    demos = list(f["data"].keys())

    print(f"Found {len(demos)} demos")

    for demo_name in tqdm(demos):

        demo = f["data"][demo_name]

        # shape: (T, H, W, 3)
        frames = demo["obs"][image_key][:]

        video_name = demo_name.replace('demo', 'episode')
        video_path = os.path.join(output_dir, f"{video_name}.mp4")

        writer = imageio.get_writer(
            video_path,
            fps=fps,
            codec="libx264",
        )

        for frame in frames:
            frame = np.flipud(frame)

            # ensure uint8
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            writer.append_data(frame)

        writer.close()

    f.close()

    print(f"Saved all videos to: {output_dir}")


if __name__ == "__main__":

    expert_dir = '/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero/data/libero_10/libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo'
    file_path = os.path.join(expert_dir, 'LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5')
    video_dir = os.path.join(expert_dir, 'videos')



    save_demo_videos(
        file_path=file_path,
        output_dir=video_dir,
        image_key="agentview_rgb",
        fps=20,
    )