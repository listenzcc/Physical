"""
File: simulation_in_long_term.py
Author: Chuncheng Zhang
Date: 2025-03-15
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Generate dynamic noise data with suddenly silent.
    Simulation the process in the long-term.

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
import matplotlib.pyplot as plt

import plotly.io as pio
import plotly.express as px

from rich import print
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import MiniBatchDictionaryLearning

from util import Subject

pio.templates.default = 'seaborn'

# %% ---- 2025-03-15 ------------------------
# Function and class


class Options:
    dt: float = 0.05  # seconds
    num_subjects: int = 40
    t_array = np.arange(0, 3600, dt)
    data_length: int = len(t_array)

    m: float = 10.5
    n: float = 0.15  # 1.5


def compute_transfer_matrix(data):
    # num of components.
    n_components = 10
    # K-means on the dictionary along the sparse code.
    n_clusters = 7

    dict_learning = MiniBatchDictionaryLearning(
        n_components=n_components, alpha=1, batch_size=200, n_jobs=8)

    # Fit the model.
    dict_learning.fit(data)

    # Get the dictionary
    dictionary = dict_learning.components_

    # Get the sparse code
    sparse_code = dict_learning.transform(data)

    # K-means on the dictionary along the sparse code
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(sparse_code)

    # Get labels
    labels = kmeans.labels_

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

    return labels, change_labels


# %% ---- 2025-03-15 ------------------------
# Play ground
opt = Options()

# Generate g_array
g_array = np.zeros((opt.data_length, opt.num_subjects))
for i in tqdm(range(opt.num_subjects)):
    subject = Subject()
    g_array[:, i] = subject.speak_willing.generate(opt.t_array + i * 1e3)
print(g_array.shape)

# Generate f_array
f_array = np.zeros((opt.data_length, opt.num_subjects))
print(f_array.shape)

# Copy from gn_array to fn_array on (0, 1) seconds
_m = opt.t_array < 1.0
f_array[_m] = g_array[_m]

# %%
# Compute
_m = opt.t_array < 1.0
_i_start = np.sum(_m)
psi_array = []
diff_array = []
for i in tqdm(range(_i_start, opt.data_length)):
    # t = opt.t_array[i]
    g = g_array[i]

    # -0.3, 0
    f1 = f_array[i-6:i]
    # -0.25, 0
    f2 = f_array[i-5:i]
    # -0.75, -0.5
    f3 = f_array[i-15: i-10]

    #
    a = np.mean(f1)
    ln = np.log1p(a) / np.log(2)

    #
    b = np.sum(f2) * opt.dt / opt.num_subjects

    #
    c = np.sum(f3) * opt.dt / opt.num_subjects

    #
    psi = 0.5 + 0.5 * np.tanh(opt.m*((b-c)+opt.n))
    # psi = 1
    psi_array.append(psi)
    diff_array.append(b-c)

    #
    h = g * ln * psi + 0.001 * g
    f_array[i] = h

    pass

# %%
a, b = np.histogram(psi_array)
sns.histplot(psi_array)
print('psi range', np.min(psi_array), np.max(psi_array))

a, b = np.histogram(diff_array)
sns.histplot(diff_array)
# Plot vline on the opt.n
plt.axvline(x=-opt.n, color='r', linestyle='--')
plt.axvline(x=opt.n, color='r', linestyle='--', label=f'n={opt.n}')
plt.legend()
print('diff range', np.min(diff_array), np.max(diff_array))
plt.show()

# %% ---- 2025-03-15 ------------------------
# Pending
m = np.mean(f_array, axis=1)
s = np.var(f_array, axis=1)

df = pd.DataFrame(m, columns=['v'])
df['name'] = 'mean'
df['t'] = opt.t_array
df2 = pd.DataFrame(s, columns=['v'])
df2['name'] = 'var'
df2['t'] = opt.t_array
df_concat = pd.concat([df, df2], axis=0)

fig = px.line(df_concat, x='t', y='v', color='name')
fig.show()

# %%
# Compute the transfer matrix and insert the transfer points into the df
# labels, change_labels = compute_transfer_matrix(g_array)
# df['change_labels'] = change_labels

# %% ---- 2025-03-15 ------------------------
# Pending
diff = np.diff(df['v'])
diff = np.concatenate([[0], diff])
df['diff'] = diff
a = np.min(diff)
b = np.max(diff)
df['scaled_diff'] = (diff - b) / (a-b)
df

# %%
# Use RdBu scheme
fig = px.scatter(df, x='t', y='v', color='diff',
                 size='scaled_diff', size_max=6,
                 opacity=df['scaled_diff'],
                 color_continuous_scale='RdBu')  # Use RdBu scheme
# Also draw the df1 mean in back
fig.add_scatter(x=df['t'], y=df['v'], mode='lines',
                name='mean', line=dict(color='grey', width=2), opacity=0.5, showlegend=False)
fig.update_traces(marker=dict(line=dict(width=0)))
fig.show()

# %%
# Mark the peak points
df['peak'] = False
found = df[df['v'] < 0.03]
for i, se in found.iterrows():
    if i-1 not in found.index:
        df.loc[i, 'peak'] = True
print(f'Found candidate peaks: {len(found)}')

lst = []
idx = 1
for i, se in df[df['peak']].iterrows():
    try:
        d = df.loc[i-100:i+100]
        k = d['v'].argmin()
        k = d.index[k]
        d = df.loc[k-50:k+100].copy()
        d['t'] = np.linspace(-50*opt.dt, 100*opt.dt, 151, endpoint=True)
        d['idx'] = idx
        idx += 1
        lst.append(d)
    except Exception as e:
        print(e)
        pass
peaks = pd.concat(lst, axis=0)
peaks

# %%
fig = px.scatter(peaks, x='t', y='v', color='diff', title='Negative peak collection',
                 color_continuous_scale='RdBu', opacity=0.8)
for i in range(peaks['idx'].max()+1):
    _df = peaks[peaks['idx'] == i]
    fig.add_scatter(x=_df['t'], y=_df['v'], mode='lines',
                    line=dict(color='grey', width=2), opacity=0.8, showlegend=False)
# Draw the annotation of change_labels on the x=t and y=1 position
# Annotate change_labels
# for i, row in peaks.iterrows():
#     if row['change_labels']:
#         fig.add_annotation(x=row['t'], y=row['v'], text=row['change_labels'],
#                            showarrow=True, arrowhead=2, ax=0, ay=-20)
fig.update_traces(marker=dict(line=dict(width=0)))

fig.show()

# %%

# %%
