"""UFLD-v2 ONNX 모델을 OpenVINO 로 돌려서 선을 그려본다. (로봇 PC 용, torch 필요 없음)

필요: pip install --user --break-system-packages openvino   (cv2, numpy 는 ROS 에 이미 있음)

예)
  # 사진 (결과는 *_lanes.jpg 로 저장)
  python3 demo_openvino.py culane_res18.onnx road.jpg field1.jpg
  # 영상 파일 / 카메라 장치 (창으로 보기, q 로 종료)
  python3 demo_openvino.py culane_res18.onnx --video example.mp4
  python3 demo_openvino.py culane_res18.onnx --video /dev/video-insta360
  # ROS 카메라 토픽 (obstacle_vision 런치를 켜둔 상태에서. 짐벌 tilt 도 런치가 맞춰줌)
  source ~/Desktop/21th_robit_intern_middle_project/colcon_ws/install/setup.bash
  python3 demo_openvino.py culane_res18.onnx --topic /camera1/camera/compressed_image

기본값은 공식 CULane 모델 기준 (선 4개, 입력 1600x320, 아래 60% 만 사용).
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import openvino as ov

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]


class LaneModel:
    def __init__(self, path, device, crop_ratio, row_lanes, col_lanes, row_anchor_top):
        core = ov.Core()
        model = core.read_model(path)
        _, _, self.in_h, self.in_w = model.input(0).get_shape()
        self.compiled = core.compile_model(model, device, {'PERFORMANCE_HINT': 'LATENCY'})
        self.req = self.compiled.create_infer_request()
        self.crop_ratio = crop_ratio
        self.row_lanes = row_lanes
        self.col_lanes = col_lanes
        self.row_anchor_top = row_anchor_top
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    def preprocess(self, bgr):
        h_resized = int(self.in_h / self.crop_ratio)
        img = cv2.resize(bgr, (self.in_w, h_resized))[h_resized - self.in_h:]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return img.transpose(2, 0, 1)[None]

    def __call__(self, bgr):
        self.req.infer({0: self.preprocess(bgr)})
        out = {o.get_any_name(): self.req.get_tensor(o).data for o in self.compiled.outputs}
        h, w = bgr.shape[:2]
        return self.decode(out, w, h)

    def decode(self, pred, img_w, img_h, local_width=1):
        """UFLD demo.py 의 pred2coords 를 numpy 로 옮긴 것."""
        lanes = []
        lanes += self._decode_axis(
            pred['loc_row'], pred['exist_row'], self.row_lanes, 2,
            np.linspace(self.row_anchor_top, 1, pred['loc_row'].shape[2]), img_w, img_h, True,
            local_width)
        lanes += self._decode_axis(
            pred['loc_col'], pred['exist_col'], self.col_lanes, 4,
            np.linspace(0, 1, pred['loc_col'].shape[2]), img_w, img_h, False, local_width)
        return lanes

    @staticmethod
    def _decode_axis(loc, exist, lane_ids, min_frac_div, anchor, img_w, img_h, is_row, lw):
        n_grid, n_cls = loc.shape[1], loc.shape[2]
        valid = exist.argmax(1)    # [1, n_cls, n_lane]
        max_idx = loc.argmax(1)    # [1, n_cls, n_lane]
        lanes = []
        for i in lane_ids:
            pts = []
            if valid[0, :, i].sum() > n_cls / min_frac_div:
                for k in range(n_cls):
                    if not valid[0, k, i]:
                        continue
                    m = int(max_idx[0, k, i])
                    ind = np.arange(max(0, m - lw), min(n_grid - 1, m + lw) + 1)
                    logits = loc[0, ind, k, i]
                    p = np.exp(logits - logits.max())
                    pos = (p / p.sum() * ind).sum() + 0.5
                    if is_row:
                        pts.append((int(pos / (n_grid - 1) * img_w), int(anchor[k] * img_h)))
                    else:
                        pts.append((int(anchor[k] * img_w), int(pos / (n_grid - 1) * img_h)))
            lanes.append((i, pts))
        return lanes


def draw(bgr, lanes, ms=None):
    out = bgr.copy()
    for i, pts in lanes:
        for p in pts:
            cv2.circle(out, p, 3, COLORS[i % len(COLORS)], -1)
    if ms is not None:
        cv2.putText(out, f'{ms:.0f} ms', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return out


def run_images(model, paths):
    for path in paths:
        bgr = cv2.imread(path)
        if bgr is None:
            print('읽기 실패:', path)
            continue
        t = time.perf_counter()
        lanes = model(bgr)
        ms = (time.perf_counter() - t) * 1000
        out_path = os.path.splitext(path)[0] + '_lanes.jpg'
        cv2.imwrite(out_path, draw(bgr, lanes))
        print(f'{path}: 선 {sum(1 for _, p in lanes if p)}개, {ms:.0f} ms -> {out_path}')


def show_loop(model, frames, save_dir):
    """frames: bgr 이미지를 계속 내주는 iterator. s = 현재 화면 저장, q = 종료"""
    n_saved = 0
    for bgr in frames:
        t = time.perf_counter()
        lanes = model(bgr)
        ms = (time.perf_counter() - t) * 1000
        cv2.imshow('ufld', draw(bgr, lanes, ms))
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            os.makedirs(save_dir, exist_ok=True)
            p = os.path.join(save_dir, f'frame_{int(time.time() * 1000)}.jpg')
            cv2.imwrite(p, bgr)  # 원본 저장 (나중에 학습 데이터로도 씀)
            n_saved += 1
            print('saved', p)
    cv2.destroyAllWindows()


def video_frames(src):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f'열기 실패: {src}')
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        yield bgr


def topic_frames(topic):
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage

    rclpy.init()
    node = rclpy.create_node('ufld_demo')
    latest = []
    node.create_subscription(
        CompressedImage, topic,
        lambda m: latest.append(np.frombuffer(m.data, np.uint8)), qos_profile_sensor_data)
    print('waiting for', topic)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)
            if not latest:
                continue
            buf = latest[-1]  # 밀린 프레임은 버리고 최신 것만
            latest.clear()
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is not None:
                yield bgr
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('onnx')
    ap.add_argument('images', nargs='*', help='사진 파일 (와일드카드 가능)')
    ap.add_argument('--video', default='', help='영상 파일 또는 카메라 장치 (/dev/video0 등)')
    ap.add_argument('--topic', default='', help='ROS CompressedImage 토픽')
    ap.add_argument('--device', default='CPU', help='CPU / GPU(인텔 내장그래픽)')
    ap.add_argument('--crop_ratio', type=float, default=0.6, help='CULane=0.6, 우리 모델=1.0')
    ap.add_argument('--row_anchor_top', type=float, default=0.42, help='CULane=0.42')
    ap.add_argument('--row_lanes', default='1,2', help='CULane: 가운데 두 선')
    ap.add_argument('--col_lanes', default='0,3', help='CULane: 바깥 두 선')
    ap.add_argument('--save_dir', default='captures', help='창에서 s 누르면 저장할 폴더')
    args = ap.parse_args()

    ids = lambda s: [int(x) for x in s.split(',') if x]  # noqa: E731
    model = LaneModel(args.onnx, args.device, args.crop_ratio,
                      ids(args.row_lanes), ids(args.col_lanes), args.row_anchor_top)
    print(f'model input {model.in_w}x{model.in_h} on {args.device}')

    if args.topic:
        show_loop(model, topic_frames(args.topic), args.save_dir)
    elif args.video:
        show_loop(model, video_frames(args.video), args.save_dir)
    else:
        paths = [p for pat in args.images for p in sorted(glob.glob(pat))]
        if not paths:
            raise SystemExit('사진, --video, --topic 중 하나는 줘야 함')
        run_images(model, paths)


if __name__ == '__main__':
    main()
