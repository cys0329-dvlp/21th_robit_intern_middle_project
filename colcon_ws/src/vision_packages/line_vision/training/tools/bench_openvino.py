"""ONNX 모델을 OpenVINO 로 돌려서 1장당 시간을 잰다. (로봇 PC 용, torch 필요 없음)

설치:  pip install openvino
예)    python3 bench_openvino.py field_384x288.onnx
       python3 bench_openvino.py field_384x288.onnx --device GPU      # 인텔 내장그래픽
       python3 bench_openvino.py *.onnx --threads 2                   # 다른 노드 몫으로 코어 남기기
"""
import argparse
import time

import numpy as np
import openvino as ov


def bench(core, path, device, threads, n):
    model = core.read_model(path)
    shape = model.input(0).get_shape()  # [1, 3, H, W]
    config = {'PERFORMANCE_HINT': 'LATENCY'}
    if device == 'CPU' and threads > 0:
        config['INFERENCE_NUM_THREADS'] = threads
    compiled = core.compile_model(model, device, config)
    req = compiled.create_infer_request()

    x = np.random.rand(*shape).astype(np.float32)
    for _ in range(10):  # 워밍업
        req.infer({0: x})

    times = []
    for _ in range(n):
        t = time.perf_counter()
        req.infer({0: x})
        times.append((time.perf_counter() - t) * 1000)
    times = np.array(times)
    print(f'{path}  [{device}]  input {shape[3]}x{shape[2]}')
    print(f'  평균 {times.mean():6.1f} ms  ({1000 / times.mean():5.1f} Hz)   '
          f'중간 {np.median(times):6.1f} ms   최악 {times.max():6.1f} ms')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('onnx', nargs='+')
    ap.add_argument('--device', default='CPU', help='CPU / GPU(인텔 내장그래픽) / AUTO')
    ap.add_argument('--threads', type=int, default=0, help='CPU 스레드 수 (0 = 전부)')
    ap.add_argument('-n', type=int, default=100, help='측정 횟수')
    args = ap.parse_args()

    core = ov.Core()
    print('OpenVINO', ov.__version__, '| 장치:', core.available_devices)
    print('CPU:', core.get_property('CPU', 'FULL_DEVICE_NAME'))
    for path in args.onnx:
        bench(core, path, args.device, args.threads, args.n)


if __name__ == '__main__':
    main()
