from huggingface_hub import HfApi

repo_id = "ducido/LIVING_ROOM_SCENE6_rollout_200eps"
local_folder = "/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero/logs/collect_data_200eps/libero_10/datacollect_diffusion_unet_libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo"

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)

api.upload_folder(
    folder_path=local_folder,
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Initial commit",
    # optional filters
    ignore_patterns=["**/__pycache__/**", "**/*.tmp", "**/.ipynb_checkpoints/**"],
    # allow_patterns=["**/*.pt","**/*.json"]  # alternatively, whitelist
)