from pathlib import Path
import shutil
from collections import defaultdict

# =========================
# ROOT DATASET FOLDER
# =========================
# root_dir = Path("data/libero_10/libero_10")
root_dir = Path("/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/lpb_libero/logs/collect_data_200eps/libero_10/datacollect_diffusion_unet_libero_10")



# =========================
# STEP 1:
# lấy toàn bộ task name từ các file .hdf5
# =========================
task_names = []

for hdf5_file in root_dir.glob("*.hdf5"):
    task_name = hdf5_file.stem
    task_names.append(task_name.replace('collect_', ''))

print(f"Found {len(task_names)} tasks")

# =========================
# STEP 2:
# tạo folder cho từng task
# =========================
for task_name in task_names:
    (root_dir / task_name).mkdir(exist_ok=True)

# =========================
# STEP 3:
# move tất cả file/folder liên quan
# =========================
all_items = list(root_dir.iterdir())

for item in all_items:

    # skip folders task vừa tạo
    if item.is_dir() and item.name in task_names:
        continue

    moved = False

    for task_name in task_names:

        # match theo substring
        if task_name in item.name:

            dst = root_dir / task_name / item.name

            print(f"MOVE: {item.name}")
            print(f"  -> {dst}")

            shutil.move(str(item), str(dst))

            moved = True
            break

    if not moved:
        print(f"SKIP: {item.name}")

print("Done!")