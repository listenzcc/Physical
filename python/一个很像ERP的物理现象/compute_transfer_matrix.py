"""
File: compute_transfer_matrix.py
Author: Chuncheng Zhang
Date: 2025-03-15
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Compute g_array's transfer matrix.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-03-15 ------------------------
# Requirements and constants
import numpy as np
import pandas as pd
import seaborn as sns

import plotly.io as pio
import plotly.express as px
import matplotlib.pyplot as plt

from rich import print
from tqdm.auto import tqdm

from util import Subject
from sklearn.cluster import KMeans
from sklearn.decomposition import MiniBatchDictionaryLearning


pio.templates.default = 'seaborn'

# %% ---- 2025-03-15 ------------------------
# Function and class


class Options:
    '''Global options'''
    dt: float = 0.05  # seconds
    num_subjects: int = 40
    t_array = np.arange(0, 3600, dt)
    data_length: int = len(t_array)

    m: float = 5
    n: float = 0.15  # 1.5


# %% ---- 2025-03-15 ------------------------
# Play ground
opt = Options()

# Generate g_array.
g_array = np.zeros((opt.data_length, opt.num_subjects))
for i in tqdm(range(opt.num_subjects)):
    subject = Subject()
    g_array[:, i] = subject.speak_willing.generate(opt.t_array + i * 1e3)
print(g_array.shape)

# %%
data = g_array

# Setup parameters
n_components = 10  # num of components
dict_learning = MiniBatchDictionaryLearning(
    n_components=n_components, alpha=1, batch_size=200, n_jobs=8)


# Fit the model.
dict_learning.fit(data)

# Get the dictionary
dictionary = dict_learning.components_

# Get the sparse code
sparse_code = dict_learning.transform(data)

print("Dictionary shape:", dictionary.shape)  # (10, 40)
print("Sparse code shape:", sparse_code.shape)  # (72000, 10)

# %%
sns.set_theme('paper')
cmap = sns.cubehelix_palette(as_cmap=True)

sns.heatmap(dictionary, cmap='RdBu')
plt.title('Dictionary')
plt.tight_layout()
plt.show()

sns.heatmap(sparse_code, cmap='RdBu')
plt.title('Sparse code')
plt.tight_layout()
plt.show()

# %%
# K-means on the dictionary along the sparse code
n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(sparse_code)

# Get labels
labels = kmeans.labels_

# (72000,)
print("K-means labels shape:", labels.shape)
# (7, 10)
print("K-means cluster centers shape:", kmeans.cluster_centers_.shape)

# Visualize the clustering.
sns.heatmap(kmeans.cluster_centers_, cmap=cmap)
plt.xlabel('Dictionary components')
plt.ylabel('Cluster centers')
plt.title('K-means Cluster Centers')
plt.tight_layout()
plt.show()

# %%
# Consider the labels are status.
# Compute status transfer matrix
status_transfer_matrix = np.zeros((n_clusters, n_clusters))
change_labels = ['' for e in labels]

for i in range(len(labels) - 1):
    current_status = labels[i]
    next_status = labels[i + 1]
    if current_status != next_status:
        status_transfer_matrix[current_status, next_status] += 1
        change_labels[i] = f'{current_status}->{next_status}'

# Normalize the transfer matrix
status_transfer_matrix = status_transfer_matrix / \
    status_transfer_matrix.sum(axis=1, keepdims=True)

# (n_clusters, n_clusters)
print("Status transfer matrix shape:", status_transfer_matrix.shape)

# Visualization of the transfer matrix
sns.heatmap(status_transfer_matrix, annot=True, fmt=".2f", cmap=cmap)
plt.title('Status Transfer Matrix')
plt.tight_layout()
plt.show()

# %%

# %%
