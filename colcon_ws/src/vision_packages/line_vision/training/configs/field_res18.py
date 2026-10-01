# 경기장 경계선용 UFLD-v2 설정 (culane_res18.py 를 바탕으로 우리 카메라에 맞게 줄임)
# 데이터/학습 관련 값은 4단계(학습)에서 다시 맞춘다. 지금은 모델 크기를 정하는 값만 의미 있음.

dataset = 'CULane'
data_root = ''  # 4단계에서 채움

# ---------------- 모델 크기 ----------------
backbone = '18'
num_lanes = 2           # 0 = 왼쪽 경계선, 1 = 오른쪽 경계선

# 카메라 640x480 (4:3) 을 그대로 줄여서 넣는다. 32 의 배수여야 함
train_width = 384
train_height = 288
crop_ratio = 1.0        # 화면 전체 사용 (카메라를 숙여서 거의 다 바닥)

# 가로줄(row anchor): 화면 위~아래에 줄 num_row 개를 긋고, 줄마다 선이 지나가는 x 를 num_cell_row 칸 중에서 고름
num_row = 36
num_cell_row = 100
# 세로줄(col anchor): 화면이 비스듬히 가로지르는 선용
num_col = 41
num_cell_col = 50

fc_norm = True
use_aux = False

# ---------------- 학습 (4단계에서 조정) ----------------
epoch = 50
batch_size = 16
optimizer = 'SGD'
learning_rate = 0.02
weight_decay = 0.0001
momentum = 0.9
scheduler = 'multi'
steps = [25, 38]
gamma = 0.1
warmup = 'linear'
warmup_iters = 100
griding_num = 200
sim_loss_w = 0.0
shp_loss_w = 0.0
mean_loss_w = 0.05
var_loss_power = 2.0
note = ''
log_path = ''
finetune = None
resume = None
test_model = ''
test_work_dir = ''
tta = False
auto_backup = True
