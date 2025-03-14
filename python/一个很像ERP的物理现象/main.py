"""
File: main.py
Author: Chuncheng Zhang
Date: 2025-03-11
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Running timeseries with NiceGUI

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-03-11 ------------------------
# Requirements and constants
import time
import numpy as np
import matplotlib.pyplot as plt

from nicegui import ui
from threading import Thread

from util import Subject, RollingData, BaseData

# %% ---- 2025-03-11 ------------------------
# Function and class
num_subjects = 20
subjects = [Subject(f'{i:04d}') for i in range(num_subjects)]
rolling_data = RollingData()


def grow():
    t = 0
    step = 0.05
    while True:
        t += step
        t = np.round(t, 2)
        for subject in subjects:
            g = subject.speak_willing.generate(t)
            bd = BaseData(t=t, g=g, name=subject.name)
            rolling_data._queue.put_nowait(bd)
        time.sleep(step)


Thread(target=grow, args=(), daemon=True).start()

label = ui.label('--:--:--')
ui_fig_1 = ui.matplotlib(figsize=(8, 4)).figure
ui_fig_2 = ui.matplotlib(figsize=(8, 4)).figure


def plot_ui():
    label.set_text(time.ctime())
    data = [e for e in rolling_data.peek()]
    x = np.array([e.t for e in data])
    x -= x[-1]
    y = np.array([e.g for e in data])
    y2 = np.array([e.s for e in data])
    name = np.array([e.name for e in data])
    names = set(name)

    ax = ui_fig_1.gca()
    ax.clear()
    with ui_fig_1:
        for n in names:
            ax.plot(x[name == n], y[name == n], '-')

    ax = ui_fig_2.gca()
    ax.clear()
    with ui_fig_2:
        for n in names:
            ax.plot(x[name == n], y2[name == n], '-')


ui.timer(1.0, lambda: plot_ui())

# %% ---- 2025-03-11 ------------------------
# Play ground
ui.run(reload=False)


# %% ---- 2025-03-11 ------------------------
# Pending


# %% ---- 2025-03-11 ------------------------
# Pending
