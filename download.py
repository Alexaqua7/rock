from huggingface_hub import snapshot_download
import os
# Official Repo for LongCLIP needed, plus the LongCLIP-L.pt from Huggingface

tt_model_name = "OpenGVLab/internimage_xl_22kto1k_384"
local_dir = './weights/OpenGVLab/internimage_xl_22kto1k_384'
os.makedirs(local_dir, exist_ok=True)
download_dir = snapshot_download(repo_id=tt_model_name, local_dir=local_dir)

tt_model_name = "OpenGVLab/internimage_l_22kto1k_384"
local_dir = './weights/OpenGVLab/internimage_l_22kto1k_384'
os.makedirs(local_dir, exist_ok=True)
download_dir = snapshot_download(repo_id=tt_model_name, local_dir=local_dir)