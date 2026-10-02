"""경기장 경계선 인식 노드.

카메라 이미지 -> UFLD 모델(OpenVINO) -> left/right 선 위 픽셀 점 -> 바닥 좌표 -> 직선
-> vision_interfaces/FieldLines (/vision/field_lines) + 디버그 이미지.
"""
import glob
import math
import os
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from vision_interfaces.msg import FieldLines

from line_vision.geometry import Camera, fit_line, ground_to_pixel, pixel_to_ground
from line_vision.lane_model import LaneModel

NAMES = ['left', 'right']
COLORS = [(0, 0, 255), (0, 255, 0)]   # BGR: left 빨강, right 초록 (라벨링 때와 같게)


class LineVisionNode(Node):
    def __init__(self):
        super().__init__('line_vision')
        p = self.declare_parameter
        image_topic = p('image_topic', '/camera1/camera/compressed_image').value
        info_topic = p('camera_info_topic', '/camera1/info').value
        out_topic = p('field_lines_topic', '/vision/field_lines').value
        debug_topic = p('debug_image_topic', '/vision/line_debug_image').value
        self.show_window = p('show_window', True).value
        image_path = p('image_path', '').value

        model_path = p('model.path', '').value or os.path.join(
            get_package_share_directory('line_vision'), 'models', 'field_best.onnx')
        device = p('model.device', 'CPU').value
        self.model = LaneModel(model_path, device, p('model.num_threads', 0).value,
                               p('model.row_anchor_top', 0.03).value)
        self.min_pixel_points = p('model.min_points', 4).value

        self.cam_height = p('camera.height', 0.60).value
        self.cam_tilt = math.radians(p('camera.tilt_deg', 27.0).value)
        self.fb = {k: p('camera.fallback.' + k, v).value for k, v in
                   dict(width=640.0, fx=471.953641, fy=476.574144, cx=309.509126, cy=228.222101).items()}
        self.use_distortion = p('camera.use_distortion', True).value

        self.max_range = p('fit.max_range', 3.0).value          # 이보다 먼 점은 버림 (멀수록 1픽셀 오차가 큼)
        self.min_fit_points = p('fit.min_points', 4).value
        self.min_length = p('fit.min_length', 0.3).value        # 바닥에서 이 길이(m) 이상 보여야 믿음
        self.outlier_m = p('fit.outlier_m', 0.05).value
        self.field_width = p('field.width', 1.4).value

        self.info = None
        self.pub = self.create_publisher(FieldLines, out_topic, 10)
        self.debug_pub = self.create_publisher(Image, debug_topic, 1)
        self.create_subscription(CameraInfo, info_topic, self.on_info, 10)

        if image_path:
            # 사진 테스트 모드: 파일 하나 또는 와일드카드 (예: /data/day1/*.jpg). 0.5초마다 다음 사진
            self.test_images = sorted(glob.glob(image_path))
            if not self.test_images:
                raise RuntimeError(f'사진을 못 찾음: {image_path}')
            self.test_i = 0
            self.create_timer(0.5, self.on_test_timer)
            self.get_logger().info(f'사진 테스트 모드: {len(self.test_images)}장')
        else:
            # 추론이 카메라보다 느리면 밀린 사진은 버리고 최신 것만 (depth 1)
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(Image, image_topic, self.on_image, qos)
        self.get_logger().info(
            f'line_vision started. model {model_path} ({self.model.in_w}x{self.model.in_h}, {device}), '
            f'image {image_topic} -> {out_topic}')

    # ------------------------------------------------------------------ 입력
    def on_info(self, msg):
        self.info = msg

    def on_image(self, msg):
        if msg.encoding not in ('bgr8', 'rgb8'):
            self.get_logger().error(f'지원 안 하는 encoding: {msg.encoding}', throttle_duration_sec=5.0)
            return
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
        img = img.reshape(msg.height, msg.width, 3)
        if msg.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.process(img, msg.header)

    def on_test_timer(self):
        path = self.test_images[self.test_i % len(self.test_images)]
        self.test_i += 1
        img = cv2.imread(path)
        if img is None:
            return
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = os.path.basename(path)
        self.process(img, msg.header)

    def camera(self, w):
        # 이미지 해상도에 맞게 K 를 맞춤 (obstacle_vision updateIntrinsics 와 같음)
        if self.info is not None and self.info.width > 0:
            s = w / self.info.width
            k = self.info.k
            dist = np.array(self.info.d, np.float64) if self.use_distortion and len(self.info.d) else None
            return Camera(k[0] * s, k[4] * s, k[2] * s, k[5] * s, self.cam_height, self.cam_tilt, dist)
        s = w / self.fb['width']
        return Camera(self.fb['fx'] * s, self.fb['fy'] * s, self.fb['cx'] * s, self.fb['cy'] * s,
                      self.cam_height, self.cam_tilt)

    # ------------------------------------------------------------------ 처리
    def process(self, img, header):
        t0 = time.perf_counter()
        h, w = img.shape[:2]
        cam = self.camera(w)
        pred = self.model.infer(img)
        t_infer = time.perf_counter() - t0
        lanes_px = self.model.decode(pred, w, h, self.min_pixel_points)

        msg = FieldLines()
        msg.header = header
        fits = []
        for name, uv in zip(NAMES, lanes_px):
            xy = pixel_to_ground(uv, cam)
            xy = xy[(xy[:, 0] > 0) & (np.hypot(xy[:, 0], xy[:, 1]) <= self.max_range)]
            f = fit_line(xy, self.outlier_m) if len(xy) >= self.min_fit_points else None
            if f is not None and (f.points < self.min_fit_points or f.length < self.min_length):
                f = None
            fits.append(f)
            if f is not None:
                setattr(msg, f'{name}_found', True)
                setattr(msg, f'{name}_dist_m', f.dist)
                setattr(msg, f'{name}_angle_deg', math.degrees(f.angle))
                setattr(msg, f'{name}_points', f.points)

        # 몸 방향: 경기장과 나란한 선이 로봇에게 +각으로 보이면 로봇은 오른쪽(-)으로 돈 것
        found = [f for f in fits if f is not None]
        if found:
            wsum = sum(f.points for f in found)
            msg.heading_valid = True
            msg.heading_deg = -math.degrees(sum(f.angle * f.points for f in found) / wsum)
        # 좌우 위치: 왼쪽 선은 경기장 y = +폭/2, 오른쪽 선은 -폭/2
        ys, ws = [], []
        if fits[0] is not None:
            ys.append(self.field_width / 2 - fits[0].dist)
            ws.append(fits[0].points)
        if fits[1] is not None:
            ys.append(-self.field_width / 2 - fits[1].dist)
            ws.append(fits[1].points)
        if ys:
            msg.field_y_valid = True
            msg.field_y_m = float(np.average(ys, weights=ws))
        if fits[0] is not None and fits[1] is not None:
            msg.field_width_m = fits[0].dist - fits[1].dist
        self.pub.publish(msg)

        ms = (time.perf_counter() - t0) * 1000
        if self.show_window or self.debug_pub.get_subscription_count() > 0:
            vis = self.draw(img, lanes_px, fits, cam, msg, ms, t_infer * 1000)
            if self.debug_pub.get_subscription_count() > 0:
                out = Image(header=header, height=vis.shape[0], width=vis.shape[1], encoding='bgr8',
                            step=vis.shape[1] * 3, data=vis.tobytes())
                self.debug_pub.publish(out)
            if self.show_window:
                cv2.imshow('line_vision', vis)
                cv2.waitKey(1)

    def draw(self, img, lanes_px, fits, cam, msg, ms, ms_infer):
        vis = img.copy()
        for i, (uv, f) in enumerate(zip(lanes_px, fits)):
            for u, v in uv.astype(int):
                cv2.circle(vis, (u, v), 3, COLORS[i], -1)
            if f is not None:
                # 맞춘 직선을 바닥에서 0.2 ~ max_range 사이로 다시 화면에 그림
                t = np.linspace(-3, 3, 61)
                xy = f.p0 + t[:, None] * f.direction
                xy = xy[(xy[:, 0] > 0.2) & (np.hypot(xy[:, 0], xy[:, 1]) <= self.max_range)]
                uv_line = ground_to_pixel(xy, cam).astype(np.int32)
                if len(uv_line) >= 2:
                    cv2.polylines(vis, [uv_line], False, (255, 255, 255), 2)
        lines = [
            f'L ' + (f'{msg.left_dist_m:+.2f}m {msg.left_angle_deg:+.0f}deg n{msg.left_points}' if msg.left_found else '-'),
            f'R ' + (f'{msg.right_dist_m:+.2f}m {msg.right_angle_deg:+.0f}deg n{msg.right_points}' if msg.right_found else '-'),
            (f'heading {msg.heading_deg:+.1f}deg' if msg.heading_valid else 'heading -')
            + (f'  field_y {msg.field_y_m:+.2f}m' if msg.field_y_valid else '')
            + (f'  width {msg.field_width_m:.2f}m' if msg.field_width_m else ''),
            f'{ms:.0f}ms (model {ms_infer:.0f})  ' + ('info:OK' if self.info is not None else 'info:fallback'),
        ]
        for j, s in enumerate(lines):
            y = 22 + 22 * j
            cv2.putText(vis, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(vis, s, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return vis


def main():
    rclpy.init()
    node = LineVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
