# line_vision 이 보내는 경기장 경계선 정보 (/vision/field_lines)
# 로봇 좌표: 원점 = 카메라 바로 아래 바닥, x 앞, y 왼쪽(+), 단위 m
std_msgs/Header header       # stamp = 촬영 시각

# 선마다: 바닥에서 직선으로 맞춘 결과
bool left_found
float32 left_dist_m          # 로봇에서 왼쪽 경계선까지 수직 거리. 선이 로봇 왼쪽에 있으면 + (보통 +)
float32 left_angle_deg       # 선 방향 - 로봇 정면 방향. 선이 반시계로 돌아 보이면 +
uint16 left_points           # 맞추는 데 쓴 점 개수 (많을수록 믿을 만함)
bool right_found
float32 right_dist_m         # 오른쪽 경계선까지. 선이 로봇 오른쪽에 있으면 - (보통 -)
float32 right_angle_deg
uint16 right_points

# 위 값으로 계산한 것 (경계선이 경기장 정면 방향과 나란하다고 가정)
bool heading_valid
float32 heading_deg          # 로봇이 경기장 정면 기준으로 틀어진 각. 왼쪽으로 돌아 있으면 +
bool field_y_valid
float32 field_y_m            # 경기장 가운데 = 0 기준 로봇 좌우 위치. 왼쪽 +. 3칸 중심 = +0.467 / 0 / -0.467 (A/3)
float32 field_width_m        # 두 선이 다 보일 때 잰 폭 (실제 1.4 와 비교해서 캘리브 확인용). 아니면 0
