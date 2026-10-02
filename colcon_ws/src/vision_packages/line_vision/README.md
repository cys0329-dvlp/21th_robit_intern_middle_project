# line_vision — 경기장 경계선 인식

카메라 이미지 → UFLD-v2 모델(OpenVINO, CPU) → 왼쪽/오른쪽 경계선 → 바닥에서 직선으로 맞춤 →
`/vision/field_lines` (`vision_interfaces/msg/FieldLines`) 로 보냄.

| 값 | 뜻 |
|---|---|
| `left_dist_m` / `right_dist_m` | 로봇에서 선까지 수직 거리. 왼쪽 선은 보통 +, 오른쪽 선은 보통 − |
| `heading_deg` | 경기장 정면 기준 로봇이 틀어진 각. 왼쪽으로 돌아 있으면 + |
| `field_y_m` | 경기장 가운데 = 0 기준 로봇 좌우 위치 (왼쪽 +). 3칸 중심 = +0.467 / 0 / −0.467 (A/3) |
| `field_width_m` | 두 선이 다 보일 때 잰 경기장 폭. **1.4 에 가까워야 카메라 설정이 맞는 것** |

학습/데이터 관련은 `training/README.md`, 전체 계획은 `PLAN.md`.

## 로봇 PC 에서 처음 돌리기

1. OpenVINO 설치 (한 번만. 1단계 A 에서 했으면 건너뜀)
   ```bash
   pip install --user --break-system-packages openvino
   python3 -c "import openvino, numpy; print(openvino.__version__, numpy.__version__)"   # numpy 는 1.26 이어야 함
   ```
2. 모델 파일: `models/field_best.onnx` (git 에 없음, 147MB). 없으면 OneDrive/USB 로 받아서 넣기
3. 빌드 (`vision_interfaces` 에 FieldLines 가 추가돼서 같이 빌드)
   ```bash
   cd ~/Desktop/21th_robit_intern_middle_project/colcon_ws
   colcon build --packages-select vision_interfaces obstacle_vision line_vision
   source install/setup.bash
   ```
4. 사진으로 먼저 확인 (카메라 없이)
   ```bash
   ros2 launch line_vision line_vision.launch.py image_path:='/경로/*.jpg'
   ```
5. 카메라로
   ```bash
   # 터미널 1: 카메라 + 장애물 (평소처럼)
   ros2 launch obstacle_vision obstacle_vision.launch.py
   # 터미널 2
   ros2 launch line_vision line_vision.launch.py
   # line_vision 만 볼 땐: ros2 launch line_vision line_vision.launch.py camera:=true
   # 터미널 3: 값 보기
   ros2 topic echo /vision/field_lines
   ```
   - 창 `line_vision`: 빨강 점 = 왼쪽 선, 초록 점 = 오른쪽 선, 흰 선 = 바닥에서 맞춘 직선,
     왼쪽 위에 값과 처리 시간(ms)
   - 모니터가 없으면 `config/line_vision.yaml` 에서 `show_window: false` →
     다른 PC 에서 `rqt_image_view /vision/line_debug_image`

## 로봇에서 확인할 것

1. **속도**: 창 왼쪽 위 `NNms (model NN)`. obstacle_vision 과 같이 켠 상태로 볼 것.
   걷기가 느려지면 `model.num_threads` 를 2~4 로.
2. **카메라 설정 맞는지**: 로봇을 경기장 **가운데, 정면**으로 세우고
   - `field_width_m` ≈ 1.4, `field_y_m` ≈ 0, `heading_deg` ≈ 0 이면 OK
   - 폭이 작게(예 1.3) 나오면 거리가 전부 짧게 계산되는 것 →
     `camera.height` / `camera.tilt_deg` / 내부값(캘리브레이션) 확인.
     **obstacle_vision.yaml 의 camera 값과 같은 원인**이니 같이 고칠 것
3. 로봇을 왼쪽 칸(+0.467), 오른쪽 칸(−0.467) 에 세워서 `field_y_m` 확인
4. 몸을 좌우로 10~20도 돌려서 `heading_deg` 부호/크기 확인
5. 판때기 뒤로 선이 가려졌을 때 이어서 잡는지

이상한 장면은 창에서 캡처하거나 `ros2 bag record /camera1/camera/compressed_image` 로 남겨두면
라벨링해서 재학습에 쓸 수 있음.
