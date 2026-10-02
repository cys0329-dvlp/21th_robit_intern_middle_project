# 장애물 1개 (같은 색 판이 옆 칸에 가까이 있으면 1개로 묶이고 count 가 2)
uint8 RED=0
uint8 BLUE=1
uint8 color

# 로봇(카메라 바로 아래 바닥) 기준 위치, 단위 m
float32 x_m        # 앞쪽 거리
float32 y_m        # 좌우, +가 왼쪽
float32 width_m    # 추정 폭. 0.40 근처면 1개, 0.80 근처면 2개가 붙은 것
uint8 count        # 붙어 있는 판 개수 추정 (width_m / 0.40)
int32 row          # 0 = 가장 가까운 행, 1, 2 ... (거리 순)

# 아랫변 상태
uint8 GROUND=0     # 바닥 접점이 보임 (거리 믿을 만함)
uint8 OCCLUDED=1   # 앞 판에 가려서 바닥 접점이 안 보임 (뒤 행 판, 거리는 행 거리로 추정)
uint8 CUT=2        # 너무 가까워서 아랫변이 화면 아래로 잘림 (거리는 실제보다 멀게 나올 수 있음)
uint8 bottom
bool cut_left      # 화면 왼쪽 끝에 걸림 -> 왼쪽으로 더 이어져 있을 수 있음 (폭/위치 부정확)
bool cut_right

# 디버그용 픽셀 값
int32 u            # 아랫변 가운데 x
int32 v            # 아랫변 y
int32 w_px         # 픽셀 폭
