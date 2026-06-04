from huggingface_hub import HfApi

repo_id = "ducido/pbrl_aloha_N30000_segM0.4_60epoch"
local_folder = "/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero/logs/pbrl/aloha_image/None/aloha_2026.06.04_06.26.50_cplkl_pseu_dpT_ExpD10_N30000_L250_1ER_SFTpos0_segM0.4_nD40_beta0.01_clip0.3_unclipwin1_smooth0.1"

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