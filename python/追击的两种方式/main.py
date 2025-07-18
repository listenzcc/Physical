"""
File: main.py
Author: Chuncheng Zhang
Date: 2025-05-30
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Pursuing the object in two ways.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-05-30 ------------------------
# Requirements and constants
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from typing import List
from vector import *

import matplotlib.pyplot as plt
import seaborn as sns

# %% ---- 2025-05-30 ------------------------
# Function and class


class MyTimer:
    dt = 10e-3  # seconds
    t = 0  # seconds

    def start(self):
        self.t = 0

    def tick(self):
        self.t += self.dt


class BaseObj:
    uid: str = 'Unique ID'  # Unique ID
    pos: tuple = (0, 0)  # position in (x, y)
    velocity: float = 1.0  # meters per second
    dir: np.ndarray = norm_vec((1, 0))  # direction vector, normalized to 1
    trace: list = []  # position list
    times: list = []  # times list
    mt: MyTimer = MyTimer()  # global timer

    def data_to_df(self):
        df = pd.DataFrame(self.trace, columns=['x', 'y'])
        df['t'] = self.times
        df['uid'] = self.uid
        df['curvature_radius'] = np.nan
        for i in tqdm(range(2, len(df)), 'Curvature radius'):
            a = tuple(df.iloc[i-2][['x', 'y']].values)
            b = tuple(df.iloc[i-1][['x', 'y']].values)
            c = tuple(df.iloc[i][['x', 'y']].values)
            r = curvature_radius(a, b, c)
            df.loc[df.index[i], 'curvature_radius'] = r
            df.loc[df.index[i], 'centripetal_acceleration'] = self.velocity ** 2 / r
        return df

    def tick(self):
        self.mt.tick()

    def step(self, dir=None):
        if dir is not None:
            self.dir = norm_vec(dir)
        v = self.velocity * self.dir * self.mt.dt
        self.pos = tuple(np.array(self.pos) + v)
        self.trace.append(self.pos)
        self.times.append(self.mt.t)


class Hit:
    hit_range: float = 1
    hit_already: bool = False
    hit_at: tuple = (0, 0, 0)


class Target(BaseObj):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.dir = norm_vec(self.dir)
        self.trace = [self.pos]
        self.times = [self.mt.t]


class Missile(BaseObj, Hit):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.dir = norm_vec(self.dir)
        self.trace = [self.pos]
        self.times = [self.mt.t]


# %% ---- 2025-05-30 ------------------------
# Play ground

# %%
# Plan 1
# Setup target and missiles
target = Target(
    uid='Target',
    pos=(0, 100),
    velocity=10,
    dir=(1, 0))

missiles: List[Missile] = [Missile(
    uid=f'M-{v}',
    pos=(20, 0),
    velocity=v,
    dir=(0, 1),
    dir1=(0-20, 100-0)
) for v in [20, 30, 40, 50]]

# Run for 1 seconds
for _ in range(int(1/target.mt.dt)):
    target.tick()
    target.step()
    for m in missiles:
        m.step()

# Run until all hit
for _ in range(50000):
    target.tick()
    target.step()
    remains = [s for s in missiles if not s.hit_already]
    if not remains:
        break
    for m in remains:
        m.step(sub_vectors(target.pos, m.pos))
        if distance(target.pos, m.pos) < m.hit_range:
            m.hit_at = (target.pos[0], target.pos[1], target.mt.t)
            m.hit_already = True
            print(f'{m} hit at {m.hit_at}')

# Summary data
df = target.data_to_df()
dfs = [m.data_to_df() for m in missiles]
dfc = pd.concat(dfs, axis=0)

# Plot
sns.set_theme('paper')
sns.lineplot(df, x='x', y='y', color='k',
             size=1, sizes=(1, 2), linewidth=0, legend=False)
sns.scatterplot(dfc, x='x', y='y', hue='uid',
                sizes=(1, 100),
                size='centripetal_acceleration',
                linewidth=0)
plt.title('Plan 1')
plt.show()


# %% ---- 2025-05-30 ------------------------
# Pending


# %% ---- 2025-05-30 ------------------------
# Pending

# %%
