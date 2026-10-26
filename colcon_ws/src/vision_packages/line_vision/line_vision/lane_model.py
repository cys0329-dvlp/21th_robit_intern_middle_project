import cv2
import numpy as np
import openvino as ov

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class LaneModel:
    def __init__(self, path, device='CPU', num_threads=0, row_anchor_top=0.03):
        core = ov.Core()
        config = {'PERFORMANCE_HINT': 'LATENCY'}
        if num_threads > 0:
            
            config['INFERENCE_NUM_THREADS'] = num_threads
        self.net = core.compile_model(core.read_model(path), device, config)
        _, _, self.in_h, self.in_w = self.net.input(0).shape
        self.row_anchor_top = row_anchor_top

    def infer(self, bgr):
        
        img = cv2.resize(bgr, (self.in_w, self.in_h))
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - MEAN) / STD
        out = self.net(img.transpose(2, 0, 1)[None])
        return {k.get_any_name(): v[0] for k, v in out.items()}

    def decode(self, pred, img_w, img_h, min_points=4, local_width=1):
        
        n_lane = pred['loc_row'].shape[2]
        row_y = np.linspace(self.row_anchor_top, 1, pred['loc_row'].shape[1]) * img_h
        col_x = np.linspace(0, 1, pred['loc_col'].shape[1]) * img_w
        lanes = []
        for i in range(n_lane):
            xs, kr = _positions(pred['loc_row'][:, :, i], pred['exist_row'][:, :, i], local_width)
            ys, kc = _positions(pred['loc_col'][:, :, i], pred['exist_col'][:, :, i], local_width)
            row_pts = np.stack([xs * img_w, row_y[kr]], 1)
            col_pts = np.stack([col_x[kc], ys * img_h], 1)
            pts = row_pts if len(row_pts) >= len(col_pts) else col_pts
            lanes.append(pts if len(pts) >= min_points else np.zeros((0, 2)))
        return lanes


def _positions(loc, exist, lw):
   
    n_grid = loc.shape[0]
    ks = np.nonzero(exist.argmax(0))[0]
    pos = np.empty(len(ks))
    for j, k in enumerate(ks):
        m = int(loc[:, k].argmax())
        ind = np.arange(max(0, m - lw), min(n_grid - 1, m + lw) + 1)
        p = np.exp(loc[ind, k] - loc[ind, k].max())
        pos[j] = ((p / p.sum() * ind).sum() + 0.5) / (n_grid - 1)
    ok = (pos >= 0) & (pos <= 1)
    return pos[ok], ks[ok]
