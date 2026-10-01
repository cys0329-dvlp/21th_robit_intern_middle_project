"""UFLD-v2 모델 -> ONNX 변환 (GPU 없이 CPU 로 동작).

--weights 를 안 주면 랜덤 가중치로 만든다. 속도는 가중치와 상관없으므로
학습 전에 "로봇에서 얼마나 빠른가" 재는 용도로 충분하다.

예)
  # 우리 설정(2선, 384x288) 랜덤 가중치 -> 속도 측정용
  python tools/export_onnx.py --ufld ufld --config configs/field_res18.py --out field_384x288.onnx
  # 입력 크기만 바꿔서
  python tools/export_onnx.py --ufld ufld --config configs/field_res18.py --width 512 --height 384 --out field_512x384.onnx
  # 공식 CULane 가중치
  python tools/export_onnx.py --ufld ufld --config ufld/configs/culane_res18.py \
      --weights weights/culane_res18.pth --out culane_res18.onnx
"""
import argparse
import os
import sys


def load_config(ufld_dir, path):
    sys.path.insert(0, os.path.abspath(ufld_dir))
    try:
        import utils.common  # noqa: F401
    except ImportError:
        # 학습용 라이브러리(DALI, tensorboard)가 없는 PC: 모델 만들 때 쓰는 건 initialize_weights 뿐이고
        # 가중치는 어차피 랜덤이거나 .pth 로 덮어쓰므로 아무것도 안 하는 함수로 대신한다
        import types
        sys.modules['utils.common'] = types.SimpleNamespace(initialize_weights=lambda *m: None)
    from utils.config import Config
    return Config.fromfile(path)


def build_model(cfg):
    # utils.common.get_model 은 .cuda() 를 부르고 torchvision 사전학습을 받으려 해서 직접 만든다
    from model.model_culane import parsingNet
    return parsingNet(
        pretrained=False, backbone=cfg.backbone,
        num_grid_row=cfg.num_cell_row, num_cls_row=cfg.num_row,
        num_grid_col=cfg.num_cell_col, num_cls_col=cfg.num_col,
        num_lane_on_row=cfg.num_lanes, num_lane_on_col=cfg.num_lanes,
        use_aux=False, input_height=cfg.train_height, input_width=cfg.train_width,
        fc_norm=cfg.fc_norm)


def load_weights(net, path):
    import torch
    try:
        state = torch.load(path, map_location='cpu')
    except Exception:  # torch 2.6+ 는 텐서 외 내용이 있으면 거부함. 공식/우리 가중치만 넣을 것
        state = torch.load(path, map_location='cpu', weights_only=False)
    state = state.get('model', state)
    state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    own = net.state_dict()
    # 모양이 다른 층(선 개수/입력 크기가 달라진 출력층)은 건너뛴다
    ok = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    skipped = [k for k in state if k not in ok]
    net.load_state_dict(ok, strict=False)
    print(f'loaded {len(ok)} tensors, skipped {len(skipped)}')
    for k in skipped:
        print('  skip', k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ufld', required=True, help='UFLD-v2 저장소 경로')
    ap.add_argument('--config', required=True)
    ap.add_argument('--weights', default='', help='.pth (없으면 랜덤)')
    ap.add_argument('--width', type=int, default=0, help='입력 폭 (0 = config 값)')
    ap.add_argument('--height', type=int, default=0, help='입력 높이 (0 = config 값)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import torch
    cfg = load_config(args.ufld, args.config)
    if args.width:
        cfg.train_width = args.width
    if args.height:
        cfg.train_height = args.height
    assert cfg.train_width % 32 == 0 and cfg.train_height % 32 == 0, '입력 크기는 32 의 배수'

    net = build_model(cfg).eval()
    if args.weights:
        load_weights(net, args.weights)

    n_param = sum(p.numel() for p in net.parameters())
    print(f'input {cfg.train_width}x{cfg.train_height}, lanes {cfg.num_lanes}, '
          f'params {n_param / 1e6:.1f}M')

    x = torch.zeros(1, 3, cfg.train_height, cfg.train_width)
    kwargs = dict(
        input_names=['input'], output_names=['loc_row', 'loc_col', 'exist_row', 'exist_col'],
        opset_version=17)
    with torch.no_grad():
        try:
            torch.onnx.export(net, x, args.out, dynamo=False, **kwargs)
        except TypeError:  # torch 2.4 이하에는 dynamo 인자가 없음
            torch.onnx.export(net, x, args.out, **kwargs)
    print('saved', args.out, f'({os.path.getsize(args.out) / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
