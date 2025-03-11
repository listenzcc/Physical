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

from util import Subject, RollingData

# %% ---- 2025-03-11 ------------------------
# Function and class
subject = Subject()
rolling_data = RollingData()


def grow():
    while True:
        t = time.time()
        g = subject.intending.generate(t)
        rolling_data.queue.put_nowait((t, g))
        time.sleep(0.05)


Thread(target=grow, args=(), daemon=True).start()

label = ui.label('--:--:--')
ui_fig = ui.matplotlib(figsize=(8, 4)).figure


def plot_ui():
    label.set_text(time.ctime())
    d = np.array(rolling_data.peek())

    ax = ui_fig.gca()
    ax.clear()
    with ui_fig:
        x = d[:, 0]
        x -= x[-1]
        y = d[:, 1]
        ax.plot(x, y, '-')


ui.timer(1.0, lambda: plot_ui())

# %% ---- 2025-03-11 ------------------------
# Play ground
ui.run(reload=False)


# %% ---- 2025-03-11 ------------------------
# Pending


# %% ---- 2025-03-11 ------------------------
# Pending
