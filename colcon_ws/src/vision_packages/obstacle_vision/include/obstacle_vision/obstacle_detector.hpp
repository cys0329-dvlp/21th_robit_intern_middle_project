#ifndef OBSTACLE_VISION__OBSTACLE_DETECTOR_HPP_
#define OBSTACLE_VISION__OBSTACLE_DETECTOR_HPP_

#include <cmath>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace obstacle_vision
{

enum Color { RED = 0, BLUE = 1 };

// 아랫변 상태
enum Bottom
{
  GROUND = 0,    // 아랫변 바로 아래가 바닥
  OCCLUDED = 1,  // 아랫변 바로 아래가 다른 판
  CUT = 2,       // 아랫변이 화면 아래 끝
};

//파라미터 

struct ColorParams
{
  // 빨강은 Hue 가 0 근처에서 한 바퀴 돌아서 두 범위를 OR 한다 (0~red_h_low, red_h_high~180)
  int red_h_low = 30;
  int red_h_high = 137;
  int red_s_min = 150;
  int red_v_min = 67;

  int blue_h_min = 80;
  int blue_h_max = 114;
  int blue_s_min = 200;
  int blue_v_min = 115;

  int morph_kernel = 5;             // open, close 커널 크기 (px)
  double min_area_ratio = 0.003;    // 화면 대비 이보다 작은 조각은 노이즈
  double split_jump_ratio = 0.02;   // 옆 열과 아랫변이 화면 높이의 이 비율 이상 차이나면 다른 판
  double min_seg_width_ratio = 0.03;  // 화면 폭 대비 이보다 좁은 조각은 버림
};

struct CameraModel
{
  double fx = 0, fy = 0, cx = 0, cy = 0;  // 현재 이미지 해상도 기준
  double cam_height = 0.45;               // 바닥 ~ 렌즈 중심 (m)
  double tilt_rad = 0.0;                  // 아래로 숙인 각이 +
};

struct GapParams
{
  double obstacle_width = 0.40;    // 장애물 실제 폭 (m)
  double row_tol_ratio = 0.25;     // 거리 차이가 (가까운 쪽 거리 x 이 값) 보다 크면 다른 행
                                   // 비율이라 캘리브레이션 스케일이 틀려도 행 구분은 유지됨
  double robot_half_width = 0.20;  // 장애물 양옆으로 이만큼 더 막힌 걸로 봄 (로봇 반폭 0.13 + 여유)
  double robot_min_half_width = 0.14;  // 여유 넣고 빈 틈이 없을 때 이것(로봇 몸 반폭 + 최소 여유)으로 한 번 더 찾음
  double search_half_width = 1.0;  // 좌우 이 범위 안에서만 빈 틈을 찾음 (m)
  double row_spacing = 1.0;        // 행 사이 거리 (m) row 0 아랫변이 잘렸을 때 row 1 거리 - 이 값으로 추정
  double near_x_m = 0.25;          // 아랫변이 잘린 판이 있을 수 있는 가장 가까운 거리 (m, 로봇 발끝 앞)
  double max_range = 6.0;          // 이보다 먼 건 무시 (m) 캘리브가 틀리면 먼 행이 멀게 나오니 여유 있게
  double min_height = 0.05;        // 보이는 높이가 이보다 낮으면 노이즈 (바닥에 붙은 테이프 등, m)
  int edge_margin_px = 3;          // 화면 끝에서 이 안이면 잘린 걸로 봄
};

// line_vision (/vision/field_lines) 이 찾은 경기장 경계선. 로봇 좌표 (x 앞, y 왼쪽 +)
struct FieldLine
{
  bool found = false;
  double dist = 0;   // 로봇에서 선까지 수직 거리, 선이 왼쪽이면 +
  double angle = 0;  // rad. 선 방향 - 로봇 정면, 반시계 +

  // 앞 x(m) 에서 이 선의 좌우 위치 y
  double yAt(double x) const {return (dist + x * std::sin(angle)) / std::cos(angle);}
};

struct FieldInfo
{
  bool valid = false;     // 최근 field_lines 를 받았고 선이 하나 이상 보임
  FieldLine left, right;
  double width = 1.4;     // 경기장 폭 A. 선이 하나만 보이면 반대쪽 선을 이만큼 떨어진 곳으로 봄
  double margin = 0.20;   // 빈 틈 목표(로봇 중심)가 선에서 이만큼은 안쪽이어야 함
};

// 결과

struct Obstacle
{
  Color color = RED;
  cv::Rect box;             // 보이는 부분 (디버그 그리기용)
  int u = 0;                // 아랫변 가운데 x
  int v = 0;                // 아랫변 y (GROUND 면 바닥 접점, OCCLUDED 면 보이는 끝)
  int w_px = 0;
  Bottom bottom = GROUND;
  int occluder_v = -1;      // OCCLUDED 일 때 가린 판의 아랫변 y (진짜 아랫변은 v 와 이것 사이)
  double occluder_x_m = 0;  // OCCLUDED 일 때 가린 판까지 거리 (진짜 거리는 이것보다 멂)
  bool cut_left = false;    // 화면 왼쪽 끝에 걸림 -> 왼쪽으로 더 이어져 있을 수 있음
  bool cut_right = false;
  double x_m = 0, y_m = 0, width_m = 0;  // 로봇 기준: 앞 +, 왼쪽 +
  int count = 1;            // 붙어 있는 판 개수 추정 (폭 / 0.40)
  int row = 0;              // 0 = 가장 가까운 행
};

struct Row
{
  double x_m = 0;           // 행 거리
  int v = -1;               // 화면에서 이 행 바닥선 y (디버그용, 없으면 -1)
};

struct Result
{
  cv::Mat red_mask, blue_mask;
  std::vector<Obstacle> obstacles;  // 행 순 (0 = 가까움), 같은 행은 왼쪽부터
  std::vector<Row> rows;
  bool detected = false;
  double nearest_x_m = 0;           // row 0 거리

  bool gap_found = false;
  double gap_y_m = 0;          // gap_found=false 여도 "이쪽으로 가보면 됨" 방향이 들어 있음
  double gap_width_m = 0;
  double view_half_width = 0;  // row 0 거리에서 화면에 보이는 좌우 반폭

  // 빈 틈 찾을 때 쓴 경기장 안쪽 범위 (field_used 면 [field_lo, field_hi] 밖은 안 고름)
  bool field_used = false;
  double field_lo = 0, field_hi = 0;
  FieldInfo field;             // 디버그 그리기용
};


//함수

// BGR -> HSV -> 빨강/파랑 마스크
void makeMasks(const cv::Mat & bgr, const ColorParams & p, cv::Mat & red, cv::Mat & blue);

// 마스크 -> 장애물 조각
std::vector<Obstacle> findSegments(
  const cv::Mat & red, const cv::Mat & blue, const ColorParams & p, int edge_margin_px);

// 바닥 위의 점 (앞 x, 왼쪽 y) <-> 화면 픽셀
bool groundToPixel(double x_m, double y_m, const CameraModel & cam, cv::Point & px);
bool pixelToGround(double u, double v, const CameraModel & cam, double & x_m, double & y_m);

// field.valid 면 경기장 밖은 빈 틈으로 고르지 않는다
Result detect(
  const cv::Mat & bgr, const ColorParams & cp, const CameraModel & cam, const GapParams & g,
  const FieldInfo & field = FieldInfo());

// 디버그 화면
cv::Mat drawResult(const cv::Mat & bgr, const Result & r, const CameraModel & cam);
cv::Mat drawMask(const Result & r);

}  // namespace obstacle_vision

#endif  // OBSTACLE_VISION__OBSTACLE_DETECTOR_HPP_
