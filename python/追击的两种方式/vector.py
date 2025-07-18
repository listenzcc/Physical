"""
File: vector.py
Author: Chuncheng Zhang
Date: 2025-05-30
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    2D vector options.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2025-05-30 ------------------------
# Requirements and constants
import math
import numpy as np
from typing import Union


# %% ---- 2025-05-30 ------------------------
# Function and class
def norm_vec(vec: Union[tuple, np.ndarray]) -> np.ndarray:
    '''Normalize the vector $vec'''
    return np.array(vec) / np.linalg.norm(vec)


def sub_vectors(a: tuple, b: tuple) -> tuple:
    '''Sub vector $a - $b'''
    return (a[0]-b[0], a[1]-b[1])


def distance(a: tuple, b: tuple) -> np.floating:
    '''Compute the distance between $a and $b'''
    return np.linalg.norm(sub_vectors(a, b))


def rotate(a: tuple, theta: float) -> tuple:
    '''Rotate vector $a with $theta radius.'''
    c = math.cos(theta)
    s = math.sin(theta)
    x, y = a
    b = (c*x-s*y, s*x+c*y)
    return b


def curvature_radius(a: tuple, b: tuple, c: tuple) -> float:
    '''
    Compute the curvature radius of the curve (a, b, c).

    :param a, tuple: The 1st point.
    :param b, tuple: The 2nd point.
    :param c, tuple: The 3rd point.

    :return radius, The radius.
    '''
    # 计算边长
    dab = math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)
    dbc = math.sqrt((c[0] - b[0])**2 + (c[1] - b[1])**2)
    dca = math.sqrt((a[0] - c[0])**2 + (a[1] - c[1])**2)

    # 计算半周长和面积
    s = (dab + dbc + dca) / 2
    area = math.sqrt(s * (s - dab) * (s - dbc) * (s - dca))

    # 避免除以零（如果三点共线，面积为0，曲率半径无限大）
    if area == 0:
        return float('inf')  # 三点共线，曲率半径无限大

    # 计算曲率半径
    radius = (dab * dbc * dca) / (4 * area)
    return radius


def curvature_radius_2(a: tuple, b: tuple, c: tuple) -> float:
    '''
    Compute the curvature radius of the curve (a, b, c).

    :param a, tuple: The 1st point.
    :param b, tuple: The 2nd point.
    :param c, tuple: The 3rd point.

    :return radius, The radius.
    '''
    ax, ay = a
    bx, by = b
    cx, cy = c

    # 计算向量 AB 和 BC
    abx = bx - ax
    aby = by - ay
    bcx = cx - bx
    bcy = cy - by

    # 计算叉积（2倍三角形面积）
    cross = abx * bcy - aby * bcx
    if cross == 0:
        return float('inf')  # 三点共线，曲率半径无限大

    # 计算边长
    dab = math.hypot(abx, aby)
    dbc = math.hypot(bcx, bcy)
    dca = math.hypot(cx - ax, cy - ay)

    # 计算曲率半径
    radius = (dab * dbc * dca) / (2 * abs(cross))
    return radius
# %% ---- 2025-05-30 ------------------------
# Play ground


# %% ---- 2025-05-30 ------------------------
# Pending


# %% ---- 2025-05-30 ------------------------
# Pending
