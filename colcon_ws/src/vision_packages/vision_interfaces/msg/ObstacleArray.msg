std_msgs/Header header       # stamp = 촬영 시각

bool detected                # 장애물이 하나라도 보이면 true
Obstacle[] obstacles         # 행 순 (row 0 = 가까움), 같은 행은 화면 왼쪽부터
float32 nearest_x_m          # 가장 가까운 행(row 0)까지 거리 (detected 가 false 면 0)
float32[] row_x_m            # 행마다 거리 (row_x_m[0] = nearest_x_m)

# 가장 가까운 행에서 로봇이 지나갈 수 있는 곳 중 정면에 제일 가까운 위치
bool gap_found
float32 gap_y_m              # +왼쪽 / -오른쪽 (m). 0 이면 그대로 직진
                             # gap_found=false 면 화면 전체가 막힌 것. 이때는 막힌 끝 중 가까운 쪽 (여기로 옆걸음)
float32 gap_width_m          # 그 빈 틈의 폭 (m)
