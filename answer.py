import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import glob

def utc_timestamp(path: str) -> float:
    return os.path.getmtime(path)

all_img_list = glob.glob('./test/*')

df = pd.DataFrame(columns=['img_path', 'rock_type'])
df['img_path'] = all_img_list
df['img_path'] = df['img_path'].apply(lambda x: x.replace('\\', '/'))
df['rock_type'] = df['img_path'].apply(lambda x : str(x).split('/')[2])
df['timestamp']   = df['img_path'].apply(utc_timestamp)

t_min, t_max = df['timestamp'].min(), df['timestamp'].max()
df['t_norm'] = (df['timestamp'] - t_min) / (t_max - t_min)

plt.figure(figsize=(12, 5))
for lbl, sub in df.groupby('rock_type'):
    plt.hist(sub['t_norm'], 60, alpha=.4, label=lbl)
plt.legend(); plt.title("Label vs. file-time histogram"); plt.xlabel("t_norm (0=earliest)")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.cluster import KMeans

def utc_timestamp(path: str) -> float:
    return os.path.getmtime(path)

# 이미지 경로 수집
all_img_list = glob.glob('./test/*')
df = pd.DataFrame({'img_path': all_img_list})
df['img_path'] = df['img_path'].apply(lambda x: x.replace('\\', '/'))
df['timestamp'] = df['img_path'].apply(utc_timestamp)

# 정규화
t_min, t_max = df['timestamp'].min(), df['timestamp'].max()
df['t_norm'] = (df['timestamp'] - t_min) / (t_max - t_min)

# 클러스터링 (예: 7개 라벨처럼 나뉠 것이라 가정)
n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['t_norm']])

# 시각화
plt.figure(figsize=(12, 5))
colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

for i in range(n_clusters):
    sub = df[df['cluster'] == i]
    plt.hist(sub['t_norm'], bins=60, alpha=0.6, color=colors[i], label=f"Cluster {i}")

plt.legend()
plt.title("Clustered histogram of test images by file timestamp")
plt.xlabel("t_norm (0 = earliest)")
plt.ylabel("Count")
plt.grid(True)
plt.show()