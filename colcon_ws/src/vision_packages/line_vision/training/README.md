# line_vision/training — AI 학습용 폴더

전체 계획은 `../PLAN.md`. 이 폴더는 colcon 빌드에서 빠진다 (`COLCON_IGNORE`).

```
training/
├── configs/field_res18.py   우리 모델 설정 (선 2개, 입력 384x288)
├── tools/
│   ├── export_onnx.py       모델 -> ONNX (CPU 로 동작)
│   ├── bench_openvino.py    ONNX 속도 측정 (로봇 PC 용, torch 필요 없음)
│   ├── demo_image.py        공식 CULane 모델(.pth)을 사진에 돌려보기 (torch)
│   └── demo_openvino.py     ONNX 모델을 사진/영상/카메라 토픽에 돌려보기 (로봇 PC 용, torch 없음)
├── weights/                 (git 에 안 올라감) .pth, .onnx
│   └── bench/               속도 측정용 ONNX (랜덤 가중치, 크기별 3개)
└── ufld/                    (git 에 안 올라감) UFLD-v2 원본 저장소를 여기에 clone
```

---

# 1단계: 준비 + 속도 확인

순서: **A (로봇 PC 속도) → B (집 컴퓨터 환경) → C (데모)**
A 를 먼저 하는 이유: 로봇에서 너무 느리면 모델 크기를 바꿔야 하니까.

## A. 로봇 PC 속도 측정 (10분)

속도는 가중치와 상관없어서, 학습 전 랜덤 모델로 재도 정확하다.

1. 노트북에서 USB 로 복사 (약 450MB):
   - `training/weights/bench/` 폴더 통째로
   - `training/tools/bench_openvino.py`
2. 로봇 PC 에서 OpenVINO 설치:
   ```bash
   pip install --user --break-system-packages openvino
   ```
   - `--break-system-packages`: Ubuntu 24.04 는 시스템 Python 에 pip 설치를 막아놔서 필요.
     나중에 ROS 노드에서 바로 import 하려면 이렇게 깔아야 한다.
   - numpy 는 기존 1.26 을 그대로 쓴다 (cv_bridge 안 깨짐). 혹시 numpy 2.x 로 올라가면 알려줄 것.
3. 측정:
   ```bash
   cd <복사한 폴더>
   python3 bench_openvino.py bench/field_320x224.onnx bench/field_384x288.onnx bench/field_512x384.onnx
   ```
4. obstacle_vision 이랑 같이 돌 때도 재본다 (다른 터미널에서 `ros2 launch obstacle_vision ...` 켜놓고 3 다시 실행)
5. 내장그래픽이 있으면 (`lspci | grep VGA` 에 Iris Xe / UHD 가 보이면):
   ```bash
   sudo apt install intel-opencl-icd
   python3 bench_openvino.py bench/field_384x288.onnx --device GPU
   ```
   CPU 를 비전/보행용으로 남겨둘 수 있어서 좋다.

6. **공식 UFLD 모델로 실제 인식까지 테스트** (torch 없이 OpenVINO 만으로 돌아감)
   USB 에 추가로 복사: `weights/culane_res18.onnx` (825MB), `data/samples/` (도로 사진/영상), `tools/demo_openvino.py`
   ```bash
   # 도로 사진 -> road_lanes.jpg 에 차선 점이 찍히면 로봇에서 UFLD 동작 OK
   python3 demo_openvino.py culane_res18.onnx samples/road.jpg
   # 도로 영상 (창으로 보기, q 종료)
   python3 demo_openvino.py culane_res18.onnx --video samples/example.mp4
   # 로봇 카메라로 경기장 보기: 다른 터미널에서 obstacle_vision 런치를 켠 뒤
   source ~/Desktop/21th_robit_intern_middle_project/colcon_ws/install/setup.bash
   python3 demo_openvino.py culane_res18.onnx --topic /camera1/camera/compressed_image
   ```
   - 카메라 창에서 **s** 를 누르면 그 장면 원본이 `captures/` 에 저장됨 → 나중에 학습 데이터로 씀
   - 공식 모델은 **도로용**이라 경기장 선은 잘 못 잡는 게 정상. 학습 전 기준점으로 보는 것
   - 공식 모델은 입력이 1600x320 이라 우리 모델(384x288)보다 4~5배 느림. 속도 판단은 위 3번 결과로 할 것

**판단 기준** (384x288 기준, 평균):
| 결과 | 다음 |
|---|---|
| 50ms 이하 | 그대로 384x288 로 진행 |
| 50~100ms | 320x224 로 내리거나 내장그래픽 사용 |
| 100ms 넘음 | 알려줄 것 (INT8 양자화 / 더 작은 모델 검토) |

참고: 개발 노트북(Ryzen AI 7 350)에서는 320x224 5.7ms / 384x288 8.3ms / 512x384 11.7ms.
i5 11세대는 이보다 2~4배 느릴 것으로 예상.

## B. 집 컴퓨터 학습 환경 (1~2시간)

UFLD-v2 학습 코드는 **Linux + NVIDIA** 전용이다 (DALI, CUDA 확장 사용).
**집 컴퓨터가 Windows 면 WSL2 Ubuntu 를 먼저 설치**한다:
PowerShell(관리자) 에서 `wsl --install -d Ubuntu-24.04` → 재부팅. 이후 명령은 전부 Ubuntu 창에서.
(Windows 용 NVIDIA 드라이버만 깔려 있으면 WSL 안에서 GPU 가 그대로 보인다)

1. GPU 확인
   ```bash
   nvidia-smi     # 오른쪽 위 "CUDA Version: 12.x" 숫자 기억
   ```
2. Miniconda 설치 (Python 환경을 프로젝트별로 분리해주는 도구)
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh     # 전부 yes, 끝나면 터미널 다시 열기
   conda create -n lane python=3.10 -y
   conda activate lane                         # 앞으로 학습할 때마다 이걸 먼저
   ```
3. PyTorch (CUDA 버전) 설치 — https://pytorch.org/get-started/locally/ 에서
   Linux / Pip / Python / CUDA(1번 숫자 이하 중 가장 높은 것) 고르면 나오는 명령 그대로. 예:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   python -c "import torch; print(torch.cuda.is_available())"    # True 나와야 함
   ```
4. 이 저장소 받고 UFLD-v2 를 training/ufld 에 clone
   ```bash
   git clone <우리 저장소 주소> ~/robit && cd ~/robit/colcon_ws/src/vision_packages/line_vision/training
   git clone https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2 ufld
   ```
5. 의존성
   ```bash
   pip install opencv-python tqdm tensorboard addict scikit-learn pathspec imagesize ujson onnx
   pip install --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda120   # CUDA 12.x 기준
   conda install -c nvidia cuda-toolkit=12.4 -y      # my_interp 빌드용 nvcc. 3번 PyTorch 의 CUDA 버전과 맞춤
   cd ufld/my_interp && pip install . --no-build-isolation && cd ../..
   ```
   (UFLD requirements.txt 의 `sklearn` 은 옛 이름이라 `scikit-learn` 으로 바꿔 넣었음)
6. 확인
   ```bash
   python -c "import torch, nvidia.dali, my_interp; print('OK')"
   ```

## C. 공식 모델 데모 (30분)

1. 사전학습 가중치 `culane_res18.pth` — 개발 노트북 `training/weights/` 에 이미 받아둠 (825MB, git 에는 없음).
   집 컴퓨터로 복사하거나 다시 받기: https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2 README 의
   CULane / ResNet18 / Google 링크 → `training/weights/culane_res18.pth`
2. 도로 사진 한 장 (인터넷 블랙박스 사진 아무거나) 과 경기장 사진 몇 장 준비
3. 실행
   ```bash
   python tools/demo_image.py --ufld ufld --weights weights/culane_res18.pth road.jpg field1.jpg field2.jpg
   ```
   → `road_lanes.jpg` 에 차선이 점으로 찍혀 있으면 **환경 OK**
   → `field*_lanes.jpg` 는 학습 전이라 엉망일 수 있음. 그게 정상. 나중에 학습 후와 비교용으로 보관.

`demo_image.py` 는 GPU 가 없어도 돌아간다 (노트북에서도 가능, 조금 느림).

---

## 1단계 끝나면 알려줄 것
- [ ] A 결과 (크기별 ms, obstacle_vision 같이 켰을 때, 내장그래픽 결과)
- [ ] 로봇 PC CPU 정확한 모델명 (A 출력 맨 위에 나옴)
- [ ] B 가 어디서 막혔는지 (막혔으면 에러 메시지 그대로)
- [ ] C 결과 사진
