"""공식 UFLD-v2 CULane 모델을 아무 사진에나 돌려서 차선을 그려본다. (GPU 없어도 됨)

1단계 확인용: 도로 사진 -> 차선이 그려지면 환경 OK
              경기장 사진 -> 학습 전에는 얼마나 못 하는지 (기준점)

예)
  python tools/demo_image.py --ufld ufld --weights weights/culane_res18.pth road.jpg field1.jpg
  -> 같은 폴더에 road_lanes.jpg, field1_lanes.jpg 저장
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_onnx import build_model, load_config, load_weights  # noqa: E402

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]


def preprocess(bgr, cfg):
    # UFLD 테스트와 같은 방식: (H/crop_ratio, W) 로 줄이고 아래쪽 H 만 잘라 씀
    h_resized = int(cfg.train_height / cfg.crop_ratio)
    img = cv2.resize(bgr, (cfg.train_width, h_resized))
    img = img[h_resized - cfg.train_height:]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return img.transpose(2, 0, 1)[None].astype(np.float32)


def decode(pred, cfg, row_lanes, col_lanes, img_w, img_h, local_width=1):
    """모델 출력 -> 선마다 (x, y) 점 목록. UFLD demo.py 의 pred2coords 와 같은 계산."""
    import torch
    row_anchor = np.linspace(0.42, 1, cfg.num_row)
    col_anchor = np.linspace(0, 1, cfg.num_col)
    lanes = []

    def soft_argmax(loc, idx, k, i, n_grid):
        all_ind = torch.arange(max(0, idx - local_width), min(n_grid - 1, idx + local_width) + 1)
        return (loc[0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5

    loc_row, exist_row = pred['loc_row'], pred['exist_row'].argmax(1)
    max_row = loc_row.argmax(1)
    n_grid_row, n_cls_row = loc_row.shape[1], loc_row.shape[2]
    for i in row_lanes:
        pts = []
        if exist_row[0, :, i].sum() > n_cls_row / 2:
            for k in range(n_cls_row):
                if exist_row[0, k, i]:
                    x = soft_argmax(loc_row, int(max_row[0, k, i]), k, i, n_grid_row)
                    pts.append((int(x / (n_grid_row - 1) * img_w), int(row_anchor[k] * img_h)))
        lanes.append((i, pts))

    loc_col, exist_col = pred['loc_col'], pred['exist_col'].argmax(1)
    max_col = loc_col.argmax(1)
    n_grid_col, n_cls_col = loc_col.shape[1], loc_col.shape[2]
    for i in col_lanes:
        pts = []
        if exist_col[0, :, i].sum() > n_cls_col / 4:
            for k in range(n_cls_col):
                if exist_col[0, k, i]:
                    y = soft_argmax(loc_col, int(max_col[0, k, i]), k, i, n_grid_col)
                    pts.append((int(col_anchor[k] * img_w), int(y / (n_grid_col - 1) * img_h)))
        lanes.append((i, pts))
    return lanes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ufld', required=True, help='UFLD-v2 저장소 경로')
    ap.add_argument('--config', default='', help='기본: <ufld>/configs/culane_res18.py')
    ap.add_argument('--weights', required=True)
    ap.add_argument('--row_lanes', default='1,2', help='CULane: 가운데 두 선은 row anchor')
    ap.add_argument('--col_lanes', default='0,3', help='CULane: 바깥 두 선은 col anchor')
    ap.add_argument('images', nargs='+')
    args = ap.parse_args()

    import torch
    cfg = load_config(args.ufld, args.config or os.path.join(args.ufld, 'configs/culane_res18.py'))
    net = build_model(cfg).eval()
    load_weights(net, args.weights)
    row_lanes = [int(s) for s in args.row_lanes.split(',') if s]
    col_lanes = [int(s) for s in args.col_lanes.split(',') if s]

    for path in args.images:
        bgr = cv2.imread(path)
        if bgr is None:
            print('읽기 실패:', path)
            continue
        h, w = bgr.shape[:2]
        with torch.no_grad():
            pred = net(torch.from_numpy(preprocess(bgr, cfg)))
        lanes = decode(pred, cfg, row_lanes, col_lanes, w, h)

        out = bgr.copy()
        found = 0
        for i, pts in lanes:
            if pts:
                found += 1
            for p in pts:
                cv2.circle(out, p, 4, COLORS[i % len(COLORS)], -1)
        out_path = os.path.splitext(path)[0] + '_lanes.jpg'
        cv2.imwrite(out_path, out)
        print(f'{path}: 선 {found}개 -> {out_path}')


if __name__ == '__main__':
    main()
