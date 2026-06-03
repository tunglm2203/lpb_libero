from huggingface_hub import HfApi

repo_id = "ducido/dp_aloha_40epoch"
local_folder = "/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero/logs/reproduce/aloha_image/None/2026.06.02_03.50.28_train_diffusion_unet_hybrid_aloha_image"

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

api.upload_folder(
    folder_path=local_folder,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Initial commit",
    # optional filters
    ignore_patterns=["**/__pycache__/**", "**/*.tmp", "**/.ipynb_checkpoints/**", "wandb/**"],
    # allow_patterns=["**/*.pt","**/*.json"]  # alternatively, whitelist
)