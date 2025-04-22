from huggingface_hub import snapshot_download
import os
# Official Repo for LongCLIP needed, plus the LongCLIP-L.pt from Huggingface

tt_model_name = "OpenGVLab/InternViT-300M-448px-V2_5"
local_dir = './weights/OpenGVLab/InternViT-300M-448px-V2_5'
os.makedirs(local_dir, exist_ok=True)
download_dir = snapshot_download(repo_id=tt_model_name, local_dir=local_dir)