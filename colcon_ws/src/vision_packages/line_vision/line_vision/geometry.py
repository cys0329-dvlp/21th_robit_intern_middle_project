"""픽셀 <-> 바닥 좌표, 바닥에서 직선 맞추기.

obstacle_vision (obstacle_detector.cpp 5번) 과 같은 카메라 모델:
  카메라 좌표 x 오른쪽, y 아래, z 앞. tilt 만큼 아래로 숙임.
  로봇 좌표 x 앞, y 왼쪽. 원점 = 카메라 바로 아래 바닥.
"""
import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float
    height: float          # m
    tilt: float            # rad, 아래로 숙인 각
    dist: np.ndarray = None  # 렌즈 왜곡 계수 (camera_info D). 없거나 0 이면 왜곡 보정 안 함


def undistort(uv, cam):
    if cam.dist is None or not np.any(cam.dist) or len(uv) == 0:
        return uv
    k = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]])
    return cv2.undistortPoints(uv.reshape(-1, 1, 2).astype(np.float64), k, cam.dist, P=k).reshape(-1, 2)


def pixel_to_ground(uv, cam):
    """(N, 2) 픽셀 -> (N, 2) 바닥 [x 앞, y 왼쪽], 수평선 위 점은 버림."""
    uv = undistort(np.asarray(uv, np.float64), cam)
    if len(uv) == 0:
        return np.zeros((0, 2))
    xn = (uv[:, 0] - cam.cx) / cam.fx
    yn = (uv[:, 1] - cam.cy) / cam.fy
    ct, st = math.cos(cam.tilt), math.sin(cam.tilt)
    denom = yn * ct + st
    ok = denom > 1e-4
    s = cam.height / denom[ok]
    return np.stack([s * (ct - yn[ok] * st), -s * xn[ok]], 1)


def ground_to_pixel(xy, cam):
    """(N, 2) 바닥 -> (N, 2) 픽셀 (왜곡 없는 모델 기준. 디버그 그림용)."""
    xy = np.asarray(xy, np.float64)
    ct, st = math.cos(cam.tilt), math.sin(cam.tilt)
    z = xy[:, 0] * ct + cam.height * st
    y = -xy[:, 0] * st + cam.height * ct
    ok = z > 1e-3
    return np.stack([cam.cx - cam.fx * xy[ok, 1] / z[ok], cam.cy + cam.fy * y[ok] / z[ok]], 1)


@dataclass
class LineFit:
    angle: float     # rad. 선 방향 - 로봇 정면. 반시계 +, (-90, 90] 도
    dist: float      # 로봇에서 선까지 수직 거리, 선이 왼쪽이면 +
    points: int      # 맞추는 데 쓴 점 (튀는 점 뺀 뒤)
    length: float    # 쓴 점들이 선 방향으로 퍼진 길이 (m)
    p0: np.ndarray   # 선 위 한 점 (점들의 평균)
    direction: np.ndarray


def fit_line(xy, outlier_m=0.05, iters=2):
    """바닥 점들에 직선 (전체 최소제곱 = 점에서 선까지 수직 거리 기준). 점이 2개 미만이면 None."""
    pts = np.asarray(xy, np.float64)
    for it in range(iters + 1):
        if len(pts) < 2:
            return None
        p0 = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - p0)
        d = vt[0]
        if d[0] < 0:   # 앞쪽을 향하게
            d = -d
        normal = np.array([-d[1], d[0]])   # 선 방향의 왼쪽
        if it == iters:
            break
        r = np.abs((pts - p0) @ normal)
        keep = r <= max(outlier_m, 3 * np.median(r))
        if keep.all():
            break
        pts = pts[keep]
    t = (pts - p0) @ d
    return LineFit(angle=math.atan2(d[1], d[0]), dist=float(p0 @ normal), points=len(pts),
                   length=float(t.max() - t.min()), p0=p0, direction=d)
