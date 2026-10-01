#include "obstacle_vision/obstacle_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace obstacle_vision
{

// ============================ 1) 색 마스크 ============================

void makeMasks(const cv::Mat & bgr, const ColorParams & p, cv::Mat & red, cv::Mat & blue)
{
  cv::Mat hsv;
  cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

  cv::Mat red1, red2;
  cv::inRange(hsv, cv::Scalar(0, p.red_s_min, p.red_v_min), cv::Scalar(p.red_h_low, 255, 255), red1);
  cv::inRange(hsv, cv::Scalar(p.red_h_high, p.red_s_min, p.red_v_min), cv::Scalar(180, 255, 255), red2);
  red = red1 | red2;

  cv::inRange(
    hsv, cv::Scalar(p.blue_h_min, p.blue_s_min, p.blue_v_min),
    cv::Scalar(p.blue_h_max, 255, 255), blue);

  // 트랙바로 빨강/파랑 범위가 겹치게 잡히면 한 픽셀이 두 색이 되니 파랑 우선으로 정리
  red.setTo(0, blue);

  // open: 작은 점 노이즈 제거, close: 판 안의 작은 구멍 메우기
  const int k = std::max(1, p.morph_kernel);
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
  for (cv::Mat * m : {&red, &blue}) {
    cv::morphologyEx(*m, *m, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(*m, *m, cv::MORPH_CLOSE, kernel);
  }
}

// ============================ 2~4) 마스크 -> 조각 ============================

namespace
{

struct Run
{
  int x, top, bot;
  Bottom st;
};

// y = a*x + b 를 이상치에 강하게 맞춘다 (튀어나온 뒤 판 열에 끌려가지 않도록)
struct Line
{
  double a = 0, b = 0;
  int inliers = 0;
  double at(double x) const {return a * x + b;}
};

// 판의 아랫변/윗변은 거의 수평 (로봇이 비스듬히 봐도 기울기 작음).
// 이보다 가파르면 판 옆 모서리 (카메라를 숙이면 세로 모서리가 비스듬히 찍힘)
constexpr double MAX_EDGE_SLOPE = 0.35;

Line robustLine(const std::vector<int> & xs, const std::vector<int> & ys, double thr)
{
  Line l;
  std::vector<int> tmp(ys);
  std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
  l.b = tmp[tmp.size() / 2];  // 시작은 중앙값 수평선

  for (int it = 0; it < 4; it++) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int n = 0, xmin = 1 << 30, xmax = -(1 << 30);
    for (size_t i = 0; i < xs.size(); i++) {
      if (std::abs(ys[i] - l.at(xs[i])) > thr) {continue;}
      sx += xs[i]; sy += ys[i]; sxx += 1.0 * xs[i] * xs[i]; sxy += 1.0 * xs[i] * ys[i];
      n++;
      xmin = std::min(xmin, xs[i]);
      xmax = std::max(xmax, xs[i]);
    }
    if (n < 2) {break;}
    const double d = n * sxx - sx * sx;
    if (xmax - xmin < 5 || std::abs(d) < 1e-9) {
      l.a = 0;
      l.b = sy / n;
    } else {
      l.a = (n * sxy - sx * sy) / d;
      l.b = (sy - l.a * sx) / n;
    }
  }
  l.inliers = 0;
  for (size_t i = 0; i < xs.size(); i++) {
    if (std::abs(ys[i] - l.at(xs[i])) <= thr) {l.inliers++;}
  }
  return l;
}

struct SegCtx
{
  Color color;
  int W, H, jump, min_w, margin;
  double min_area;
};

// 조각 -> 장애물. 아랫변이 판 아랫변답지 않으면 (가파름) false
bool makeObstacle(const std::vector<Run> & runs, const SegCtx & c, Obstacle & o)
{
  std::vector<int> xs, bots;
  for (const auto & r : runs) {
    xs.push_back(r.x);
    bots.push_back(r.bot);
  }
  // 아랫변은 직선으로 맞춤 (로봇이 비스듬히 봐서 아랫변이 기울어도 됨)
  const double thr = std::max(2, c.jump / 2);
  const Line bl = robustLine(xs, bots, thr);
  if (std::abs(bl.a) > MAX_EDGE_SLOPE || bl.inliers < c.min_w) {return false;}

  // 좌우 범위 = 아랫변 직선 위에 있는 열만. 카메라를 숙이면 판 위쪽이 옆으로 벌어져 찍히는데
  // 그 부분(열의 아래 끝이 옆 모서리에서 끝남)은 폭에 넣지 않는다
  int x0 = 1 << 30, x1 = -1;
  std::vector<int> tops;
  for (const auto & r : runs) {
    if (std::abs(r.bot - bl.at(r.x)) > thr) {continue;}
    x0 = std::min(x0, r.x);
    x1 = std::max(x1, r.x);
    tops.push_back(r.top);
  }
  if (x1 - x0 + 1 < c.min_w) {return false;}

  o = Obstacle();
  o.color = c.color;
  o.u = (x0 + x1) / 2;
  o.v = static_cast<int>(std::lround(bl.at(o.u)));
  std::nth_element(tops.begin(), tops.begin() + tops.size() / 2, tops.end());
  const int top = std::min(tops[tops.size() / 2], o.v);
  o.box = cv::Rect(x0, top, x1 - x0 + 1, std::max(1, o.v - top + 1));
  o.w_px = x1 - x0 + 1;

  // 아랫변 상태는 다수결
  int cnt[3] = {0, 0, 0};
  for (const auto & r : runs) {cnt[r.st]++;}
  o.bottom = static_cast<Bottom>(std::max_element(cnt, cnt + 3) - cnt);

  o.cut_left = x0 <= c.margin;
  o.cut_right = x1 >= c.W - 1 - c.margin;
  return true;
}

// 한 조각 처리: 같은 색 뒤 판이 위로 튀어나와 있으면 떼어내서 따로 처리 (재귀, 행 3개라 깊이 2 까지)
void processSegment(
  const std::vector<Run> & runs, const SegCtx & c, int depth, std::vector<Obstacle> & out)
{
  if (runs.empty()) {return;}
  if (runs.back().x - runs.front().x + 1 < c.min_w) {return;}
  double area = 0;
  for (const auto & r : runs) {area += r.bot - r.top + 1;}
  if (area < c.min_area) {return;}

  std::vector<int> xs, tops;
  for (const auto & r : runs) {
    xs.push_back(r.x);
    tops.push_back(r.top);
  }
  const Line top_line = robustLine(xs, tops, std::max(2, c.jump / 2));

  std::vector<Run> base(runs);
  // 뒤 판 떼어내기는 이 판의 윗변이 확실할 때만:
  //   - 윗변이 화면 위 끝에 붙어 있지 않음 (판이 화면보다 크면 뒤 판이 위로 보일 수 없음)
  //   - 윗변이 거의 수평이고 과반수 열이 그 직선 위 (옆 모서리가 비스듬한 걸 윗변으로 착각하지 않게)
  const bool top_cut = top_line.at((xs.front() + xs.back()) / 2.0) <= c.margin;
  const bool top_ok = std::abs(top_line.a) <= MAX_EDGE_SLOPE &&
    top_line.inliers * 2 >= static_cast<int>(xs.size());
  if (!top_cut && top_ok && depth < 2) {
    // 윗변 직선보다 jump 이상 위로 솟은 열들 = 뒤 판 (아랫변이 이 판 윗변에 가려짐)
    std::vector<Run> up;
    auto flush = [&]() {
        if (!up.empty() && up.back().x - up.front().x + 1 >= c.min_w) {
          processSegment(up, c, depth + 1, out);
        }
        up.clear();
      };
    for (auto & r : base) {
      const int line_y = static_cast<int>(std::lround(top_line.at(r.x)));
      if (r.top < line_y - c.jump) {
        if (!up.empty() && r.x - up.back().x > 2) {flush();}
        up.push_back({r.x, r.top, line_y - 1, OCCLUDED});
        r.top = line_y;
      }
    }
    flush();
  }
  Obstacle o;
  if (makeObstacle(base, c, o)) {out.push_back(o);}
}

}  // namespace

std::vector<Obstacle> findSegments(
  const cv::Mat & red, const cv::Mat & blue, const ColorParams & p, int edge_margin_px)
{
  std::vector<Obstacle> out;
  const int H = red.rows;
  const int W = red.cols;
  const int jump = std::max(2, static_cast<int>(p.split_jump_ratio * H));
  const int look = std::max(3, jump / 2);  // 아랫변 바로 아래 이 픽셀 안에 판이 있으면 가려진 것

  // 열 단위로 훑기 쉽게 전치 (행 = 원래 열)
  cv::Mat rt = red.t();
  cv::Mat bt = blue.t();

  for (Color color : {RED, BLUE}) {
    const cv::Mat & mt = (color == RED) ? rt : bt;

    // (1) 열마다 구간 찾기
    std::vector<std::vector<Run>> cols(W);
    for (int x = 0; x < W; x++) {
      const uchar * m = mt.ptr<uchar>(x);
      const uchar * ra = rt.ptr<uchar>(x);
      const uchar * ba = bt.ptr<uchar>(x);
      int y = 0;
      while (y < H) {
        if (!m[y]) {y++; continue;}
        const int top = y;
        while (y < H && m[y]) {y++;}
        const int bot = y - 1;
        if (bot - top + 1 < 2) {continue;}

        Bottom st = GROUND;
        if (bot >= H - 1 - edge_margin_px) {
          st = CUT;
        } else {
          for (int yy = bot + 1; yy <= std::min(H - 1, bot + look); yy++) {
            if (ra[yy] || ba[yy]) {st = OCCLUDED; break;}
          }
        }
        cols[x].push_back({x, top, bot, st});
      }
    }

    // (2) 옆 열 구간끼리 잇기: 같은 아랫변 상태, 세로로 겹치고, 아랫변 차이 jump 이하
    //     한 구간은 오른쪽 구간 하나에만 이어진다 (사슬)
    std::vector<std::vector<int>> id(W);
    std::vector<std::vector<Run>> chains;
    for (int x = 0; x < W; x++) {
      id[x].assign(cols[x].size(), -1);
      std::vector<bool> taken[2];
      for (int back = 1; back <= 2; back++) {
        if (x - back >= 0) {taken[back - 1].assign(cols[x - back].size(), false);}
      }
      for (size_t j = 0; j < cols[x].size(); j++) {
        const Run & b = cols[x][j];
        int best_back = 0, best_i = -1, best_d = jump + 1;
        for (int back = 1; back <= 2 && best_i < 0; back++) {  // 한 열 빠져도 이어줌
          if (x - back < 0) {break;}
          const auto & prev = cols[x - back];
          for (size_t i = 0; i < prev.size(); i++) {
            const Run & a = prev[i];
            if (taken[back - 1][i] || a.st != b.st) {continue;}
            if (std::min(a.bot, b.bot) < std::max(a.top, b.top)) {continue;}
            const int d = std::abs(a.bot - b.bot);
            if (d < best_d && id[x - back][i] >= 0) {
              best_d = d;
              best_i = static_cast<int>(i);
              best_back = back;
            }
          }
        }
        if (best_i >= 0) {
          taken[best_back - 1][best_i] = true;
          id[x][j] = id[x - best_back][best_i];
        } else {
          id[x][j] = static_cast<int>(chains.size());
          chains.emplace_back();
        }
        chains[id[x][j]].push_back(b);
      }
    }

    // (3) 사슬 하나 = 조각 하나
    SegCtx c{color, W, H, jump,
      std::max(2, static_cast<int>(p.min_seg_width_ratio * W)), edge_margin_px,
      p.min_area_ratio * W * H};
    for (const auto & ch : chains) {
      processSegment(ch, c, 0, out);
    }
  }
  return out;
}

// ============================ 5) 픽셀 <-> m ============================
//
// 카메라 좌표: x 오른쪽, y 아래, z 앞. tilt 만큼 아래로 숙어 있음.
// 로봇 좌표: 앞(forward), 왼쪽(left), 위(up). 원점은 카메라 바로 아래 바닥.
//   forward = z*cos(t) - y*sin(t)
//   up      = -(y*cos(t) + z*sin(t))
//   left    = -x

bool pixelToGround(double u, double v, const CameraModel & cam, double & x_m, double & y_m)
{
  const double xn = (u - cam.cx) / cam.fx;
  const double yn = (v - cam.cy) / cam.fy;
  const double ct = std::cos(cam.tilt_rad);
  const double st = std::sin(cam.tilt_rad);
  const double denom = yn * ct + st;  // 광선이 아래로 향하는 정도 (<=0 이면 수평선 위)
  if (denom <= 1e-4) {return false;}
  const double s = cam.cam_height / denom;  // 카메라 z 방향 깊이
  x_m = s * (ct - yn * st);
  y_m = -s * xn;
  return true;
}

bool groundToPixel(double x_m, double y_m, const CameraModel & cam, cv::Point & px)
{
  const double ct = std::cos(cam.tilt_rad);
  const double st = std::sin(cam.tilt_rad);
  const double up = -cam.cam_height;
  const double z = x_m * ct - up * st;
  const double y = -x_m * st - up * ct;
  const double x = -y_m;
  if (z <= 1e-3) {
    return false;
  }
  px = cv::Point(
    static_cast<int>(cam.cx + cam.fx * x / z),
    static_cast<int>(cam.cy + cam.fy * y / z));
  return true;
}

namespace
{

// 앞 거리 x 가 정해졌을 때 좌우 위치, 폭
void setAtDistance(Obstacle & o, double x_m, const CameraModel & cam)
{
  const double z = x_m * std::cos(cam.tilt_rad) + cam.cam_height * std::sin(cam.tilt_rad);
  o.x_m = x_m;
  o.y_m = -z * (o.u - cam.cx) / cam.fx;
  o.width_m = o.w_px * z / cam.fx;
}

double depthAt(double x_m, const CameraModel & cam)
{
  return x_m * std::cos(cam.tilt_rad) + cam.cam_height * std::sin(cam.tilt_rad);
}

// ============================ 6) 행 ============================

void assignRows(Result & r, const GapParams & g)
{
  std::vector<Row> & rows = r.rows;
  const double tol = g.row_tol_ratio;

  // (1) 바닥 접점이 보이는 판 (GROUND, CUT): 가까운 순으로, 행 첫 판보다 tol 비율 이상 멀면 새 행
  std::vector<Obstacle *> vis, occ;
  for (auto & o : r.obstacles) {(o.bottom == OCCLUDED ? occ : vis).push_back(&o);}
  std::sort(vis.begin(), vis.end(), [](auto * a, auto * b) {return a->x_m < b->x_m;});
  std::vector<std::vector<double>> members;
  double row_start = 0;
  for (auto * o : vis) {
    if (rows.empty() || o->x_m > row_start * (1 + tol)) {
      rows.push_back({});
      members.emplace_back();
      row_start = o->x_m;
    }
    members.back().push_back(o->x_m);
    o->row = static_cast<int>(rows.size()) - 1;
  }
  for (size_t i = 0; i < rows.size(); i++) {  // 행 거리 = 중앙값
    auto & m = members[i];
    std::nth_element(m.begin(), m.begin() + m.size() / 2, m.end());
    rows[i].x_m = m[m.size() / 2];
  }

  // (2) 가려진 판: 진짜 거리는 (가린 판 거리, 보이는 끝 거리] 사이.
  //     그 사이에 이미 있는 행이 있으면 그 행 (가까운 쪽), 없으면 보이는 끝 거리로 새 행
  std::sort(occ.begin(), occ.end(), [](auto * a, auto * b) {return a->x_m < b->x_m;});
  for (auto * o : occ) {
    const double x_hi = o->x_m;  // detect() 에서 보이는 끝 거리를 넣어 둠
    const double x_lo = o->occluder_x_m;  // 가린 판의 행보다는 확실히 멀어야 함
    int pick = -1;
    for (size_t i = 0; i < rows.size(); i++) {
      if (rows[i].x_m > x_lo * (1 + tol) && rows[i].x_m <= x_hi * (1 + tol) &&
        (pick < 0 || rows[i].x_m < rows[pick].x_m))
      {
        pick = static_cast<int>(i);
      }
    }
    if (pick < 0) {
      rows.push_back({x_hi, -1});
      pick = static_cast<int>(rows.size()) - 1;
    }
    o->row = pick;
  }

  // (3) 행 번호를 가까운 순으로
  std::vector<int> order(rows.size());
  for (size_t i = 0; i < order.size(); i++) {order[i] = static_cast<int>(i);}
  std::sort(order.begin(), order.end(), [&](int a, int b) {return rows[a].x_m < rows[b].x_m;});
  std::vector<int> rank(rows.size());
  std::vector<Row> sorted;
  for (size_t i = 0; i < order.size(); i++) {
    rank[order[i]] = static_cast<int>(i);
    sorted.push_back(rows[order[i]]);
  }
  rows = sorted;
  for (auto & o : r.obstacles) {o.row = rank[o.row];}
}

// ============================ 7) 같은 행 조각 합치기 ============================
// 같은 행, 같은 색, 화면에서 붙어 있음 = 한 판이 앞 판에 일부 가려져 둘로 보이거나, 옆 칸 판 2개가 붙은 것.
// 경기장 칸 간격(0.425m) ~= 판 폭(0.43m) 이라 같은 행 판끼리는 딱 붙어 있거나 한 칸(판 1개 폭) 떨어져 있다.
// 그래서 판 폭보다 한참 작은 간격(화면 폭 4%)까지는 같은 판/붙은 판으로 합쳐도 된다.

// 두 조각 사이가 아랫변 조금 위에서 같은 색으로 꽉 차 있으면 한 판 (아랫변에 홈이 파인 경우).
// 너무 위를 보면, 한 칸 떨어진 같은 색 판 두 개 사이로 보이는 같은 색 뒤 판 때문에 빈 칸이 막힌 걸로 합쳐짐
bool bridged(const Result & r, const Obstacle & a, const Obstacle & b)
{
  const cv::Mat & m = (a.color == RED) ? r.red_mask : r.blue_mask;
  const int bot = std::min(a.v, b.v);
  const int y = bot - (bot - std::max(a.box.y, b.box.y)) / 4;
  const int x0 = std::min(a.box.x + a.box.width, b.box.x + b.box.width);
  const int x1 = std::max(a.box.x, b.box.x);
  if (y < 0 || y >= m.rows || x1 <= x0) {return false;}
  return cv::countNonZero(m.row(y).colRange(x0, x1)) >= 0.9 * (x1 - x0);
}

void mergeRow(Result & r, const CameraModel & cam, int img_w)
{
  const int max_gap = std::max(3, img_w / 25);
  bool merged = true;
  while (merged) {
    merged = false;
    for (size_t i = 0; i < r.obstacles.size() && !merged; i++) {
      for (size_t j = i + 1; j < r.obstacles.size() && !merged; j++) {
        Obstacle & a = r.obstacles[i];
        const Obstacle & b = r.obstacles[j];
        if (a.row != b.row || a.color != b.color) {continue;}
        const int gap = std::max(a.box.x, b.box.x) -
          std::min(a.box.x + a.box.width, b.box.x + b.box.width);
        if (gap > max_gap && !bridged(r, a, b)) {continue;}

        // 거리/아랫변은 바닥 접점이 보이는 쪽 것을 쓴다
        double x;
        if ((a.bottom == OCCLUDED) != (b.bottom == OCCLUDED)) {
          const Obstacle & vis = (a.bottom == OCCLUDED) ? b : a;
          x = vis.x_m;
          a.v = vis.v;
          a.bottom = vis.bottom;
        } else {
          x = std::min(a.x_m, b.x_m);
          a.v = std::max(a.v, b.v);
          if (b.bottom == CUT) {a.bottom = CUT;}
        }
        a.box = a.box | b.box;
        a.cut_left = a.cut_left || b.cut_left;
        a.cut_right = a.cut_right || b.cut_right;
        a.u = a.box.x + a.box.width / 2;
        a.w_px = a.box.width;
        setAtDistance(a, x, cam);
        r.obstacles.erase(r.obstacles.begin() + j);
        merged = true;
      }
    }
  }
}

// 한쪽 옆이 앞 행 판이나 화면 끝에 가려 폭이 판 1개보다 좁게 보이면, 보이는 쪽 모서리는 그대로 두고
// 가려진 쪽으로 판 1개 폭까지 늘린다 (보이는 부분 가운데를 판 가운데로 보내면 위치가 틀어지므로)
void completeHidden(Result & r, const GapParams & g, int img_w)
{
  const int max_gap = std::max(3, img_w / 100);
  // 아랫변 바로 위 높이에서 옆 max_gap 안에 다른 판 픽셀이 있으면 그쪽이 가려진 것.
  // (그 높이까지 내려와 있는 판 = 이 판보다 가깝거나 같은 행. 같은 행이면 폭이 이미 판 1개라 여기 안 옴)
  auto touching = [&](const Obstacle & o, int dir) {
      const int y = std::max(0, o.v - 3);
      const int xs = (dir < 0) ? o.box.x - 1 : o.box.x + o.box.width;
      for (int k = 0; k < max_gap; k++) {
        const int x = xs + dir * k;
        if (x < 0 || x >= img_w) {return false;}
        if (r.red_mask.at<uchar>(y, x) || r.blue_mask.at<uchar>(y, x)) {return true;}
      }
      return false;
    };
  for (auto & o : r.obstacles) {
    if (o.width_m >= 0.9 * g.obstacle_width) {continue;}
    // 화면 끝에 걸린 쪽이 확실한 쪽 (반대편에 닿은 건 같은 행 옆 판일 수 있음)
    bool hl = o.cut_left, hr = o.cut_right;
    if (!hl && !hr) {
      hl = touching(o, -1);
      hr = touching(o, +1);
    }
    if (hl == hr) {continue;}  // 양쪽 다 가려지면 어디로 늘릴지 모름
    if (hl) {  // 화면 왼쪽 = +y
      o.y_m = (o.y_m - o.width_m / 2) + g.obstacle_width / 2;
    } else {
      o.y_m = (o.y_m + o.width_m / 2) - g.obstacle_width / 2;
    }
    o.width_m = g.obstacle_width;
  }
}

// ============================ 8) 빈 틈 ============================

void findGap(Result & r, const CameraModel & cam, const GapParams & g, int img_w)
{
  r.detected = !r.obstacles.empty();
  if (!r.detected) {
    // 앞에 아무것도 없음 -> 그대로 직진
    r.gap_found = true;
    r.gap_y_m = 0;
    r.gap_width_m = 2 * g.search_half_width;
    return;
  }
  r.nearest_x_m = r.rows[0].x_m;

  // row 0 장애물이 막고 있는 좌우 구간 (로봇 반폭만큼 넓혀서).
  // 화면 끝에 걸린 판은 화면 밖으로 계속 이어진다고 본다
  constexpr double INF = 1e3;
  std::vector<std::pair<double, double>> blocked;
  for (const auto & o : r.obstacles) {
    if (o.row != 0) {continue;}
    const double half = o.width_m / 2 + g.robot_half_width;
    double lo = o.y_m - half, hi = o.y_m + half;
    if (o.cut_left) {hi = INF;}    // 화면 왼쪽 = +y
    if (o.cut_right) {lo = -INF;}
    blocked.push_back({lo, hi});
  }

  // 화면 밖은 모르는 곳이라 빈 곳으로 치지 않는다
  const double half_fov_tan = (img_w / 2.0) / cam.fx;
  r.view_half_width = std::min(g.search_half_width, depthAt(r.nearest_x_m, cam) * half_fov_tan);
  const double lo = -r.view_half_width;
  const double hi = r.view_half_width;

  // 막힌 구간 합치기 -> 사이사이가 빈 구간
  std::sort(blocked.begin(), blocked.end());
  std::vector<std::pair<double, double>> free_iv;
  double cur = lo;
  for (const auto & b : blocked) {
    if (b.first > cur) {
      free_iv.push_back({cur, std::min(b.first, hi)});
    }
    cur = std::max(cur, b.second);
    if (cur >= hi) {break;}
  }
  if (cur < hi) {
    free_iv.push_back({cur, hi});
  }

  // 보이는 범위가 전부 막혔으면: 막힌 덩어리의 좌/우 끝 중 가까운 쪽을 가리킨다
  // (한 행에 최대 2개라 옆걸음하면 빈 칸이 보이기 시작함)
  const double edge_l = std::max_element(blocked.begin(), blocked.end(),
      [](const auto & a, const auto & b) {return a.second < b.second;})->second;
  const double edge_r = blocked.front().first;
  const bool l_ok = edge_l < INF / 2, r_ok = edge_r > -INF / 2;
  if (l_ok && (!r_ok || std::abs(edge_l) < std::abs(edge_r))) {
    r.gap_y_m = edge_l;
  } else if (r_ok) {
    r.gap_y_m = edge_r;
  } else {
    r.gap_y_m = 0;  // 양쪽 다 화면 밖까지 막힘 -> 한 발 물러서서 다시 봐야 함
  }
  r.gap_width_m = 0;

  // 빈 구간 중 정면(0)에서 가장 가까운 점 = 옆걸음이 제일 적게 드는 곳
  r.gap_found = false;
  double best = 1e9;
  for (const auto & f : free_iv) {
    if (f.second <= f.first) {continue;}
    const double target = std::clamp(0.0, f.first, f.second);
    if (std::abs(target) < std::abs(best)) {
      best = target;
      r.gap_found = true;
      r.gap_y_m = target;
      r.gap_width_m = f.second - f.first;
    }
  }
}

}  // namespace

// ============================ 전체 ============================

Result detect(
  const cv::Mat & bgr, const ColorParams & cp, const CameraModel & cam, const GapParams & g)
{
  Result r;
  const int H = bgr.rows;
  const int W = bgr.cols;
  makeMasks(bgr, cp, r.red_mask, r.blue_mask);
  std::vector<Obstacle> segs = findSegments(r.red_mask, r.blue_mask, cp, g.edge_margin_px);

  // 가려진 조각: 바로 아래에서 가리고 있는 판의 아랫변 찾기 (가로로 가장 많이 겹치는 것)
  const int look = std::max(3, static_cast<int>(cp.split_jump_ratio * H) / 2) + 2;
  for (auto & o : segs) {
    if (o.bottom != OCCLUDED) {continue;}
    int best = 0;
    o.occluder_v = H - 1;
    for (const auto & f : segs) {
      if (&f == &o || f.v <= o.v || f.box.y > o.v + look) {continue;}
      const int ov = std::min(o.box.x + o.box.width, f.box.x + f.box.width) -
        std::max(o.box.x, f.box.x);
      if (ov > best) {
        best = ov;
        o.occluder_v = f.v;
      }
    }
  }

  // 아랫변 바로 아래에서 양옆이 같은 색이면 바닥이 아니라 앞 판에 난 구멍/홈 (스티커, 뜯긴 모서리 등).
  // 진짜 뒤 판이면 양옆이 같은 색 앞 판일 수 없다 (같은 행 판끼리는 붙어 있어서 사이로 뒤가 안 보임)
  const int side = std::max(3, W / 100);
  auto isHole = [&](const Obstacle & o) {
      const cv::Mat & m = (o.color == RED) ? r.red_mask : r.blue_mask;
      const int y = std::min(H - 1, o.v + 3);
      bool l = false, rr = false;
      for (int k = 1; k <= side; k++) {
        const int xl = o.box.x - k, xr = o.box.x + o.box.width - 1 + k;
        if (xl >= 0 && m.at<uchar>(y, xl)) {l = true;}
        if (xr < W && m.at<uchar>(y, xr)) {rr = true;}
      }
      return l && rr;
    };

  // 픽셀 -> m, 말이 안 되는 것 거르기
  const double ct = std::cos(cam.tilt_rad);
  const double st = std::sin(cam.tilt_rad);
  for (auto & o : segs) {
    if (o.bottom == GROUND && isHole(o)) {continue;}
    double x = 0, y = 0;
    if (o.bottom == GROUND) {
      if (!pixelToGround(o.u, o.v, cam, x, y)) {continue;}  // 수평선 위에 아랫변 = 바닥에 선 판이 아님
    } else if (o.bottom == CUT) {
      // 바닥 접점은 화면 아래 끝보다 가깝다. 폭으로 잰 거리(판 1개 가정)와 비교해 가까운 쪽
      if (!pixelToGround(o.u, H - 1, cam, x, y)) {continue;}
      if (!o.cut_left && !o.cut_right) {
        const double xw = (cam.fx * g.obstacle_width / o.w_px - cam.cam_height * st) / ct;
        if (xw > 0) {x = std::min(x, xw);}
      }
    } else {
      // 가려짐: 진짜 거리는 보이는 끝 거리보다 가깝다. 일단 보이는 끝 거리 (행 나눌 때 다시 정함)
      if (!pixelToGround(o.u, o.v, cam, x, y)) {x = g.max_range;}
      if (pixelToGround(o.u, o.occluder_v, cam, o.occluder_x_m, y) && o.occluder_x_m > g.max_range) {
        continue;
      }
    }
    if (o.bottom != OCCLUDED && x > g.max_range) {continue;}
    setAtDistance(o, x, cam);

    // 보이는 높이가 너무 낮으면 (바닥 테이프, 선 반사 등) 버림. 가려진 판은 조금만 보일 수 있어서 제외
    const double h_m = (o.v - o.box.y) * depthAt(x, cam) / cam.fy;
    if (o.bottom != OCCLUDED && o.box.y > g.edge_margin_px && h_m < g.min_height) {continue;}

    r.obstacles.push_back(o);
  }

  if (!r.obstacles.empty()) {
    assignRows(r, g);
    for (auto & o : r.obstacles) {
      if (o.bottom == OCCLUDED) {setAtDistance(o, r.rows[o.row].x_m, cam);}
    }
    mergeRow(r, cam, W);
    completeHidden(r, g, W);
    for (auto & o : r.obstacles) {
      o.count = std::max(1, static_cast<int>(std::lround(o.width_m / g.obstacle_width)));
    }
    std::sort(
      r.obstacles.begin(), r.obstacles.end(), [](const Obstacle & a, const Obstacle & b) {
        return a.row != b.row ? a.row < b.row : a.box.x < b.box.x;
      });

    // 행 바닥선 (디버그): 바닥 접점이 보이는 판 아랫변 중앙값, 없으면 행 거리를 투영
    for (size_t i = 0; i < r.rows.size(); i++) {
      std::vector<int> vs;
      for (const auto & o : r.obstacles) {
        if (o.row == static_cast<int>(i) && o.bottom == GROUND) {vs.push_back(o.v);}
      }
      if (!vs.empty()) {
        std::nth_element(vs.begin(), vs.begin() + vs.size() / 2, vs.end());
        r.rows[i].v = vs[vs.size() / 2];
      } else {
        cv::Point p;
        r.rows[i].v = groundToPixel(r.rows[i].x_m, 0, cam, p) ? p.y : -1;
      }
    }
  }

  findGap(r, cam, g, W);
  return r;
}

// ============================ 디버그 그리기 ============================

static void putLabel(cv::Mat & img, const std::string & text, cv::Point org, cv::Scalar color, double scale)
{
  const int th = std::max(1, static_cast<int>(scale * 2));
  cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(0, 0, 0), th + 2);
  cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, color, th);
}

static const char * bottomTag(Bottom b)
{
  return b == GROUND ? "G" : (b == CUT ? "C" : "O");
}

cv::Mat drawResult(const cv::Mat & bgr, const Result & r, const CameraModel & cam)
{
  cv::Mat img = bgr.clone();
  const double sc = std::max(0.4, img.cols / 1000.0);
  char buf[128];

  // 행마다 흰 점선 (바닥선)
  for (size_t i = 0; i < r.rows.size(); i++) {
    const int y = r.rows[i].v;
    if (y < 0 || y >= img.rows) {continue;}
    for (int x = 0; x < img.cols; x += 20) {
      cv::line(img, {x, y}, {std::min(x + 10, img.cols - 1), y}, cv::Scalar(255, 255, 255), 2);
    }
    std::snprintf(buf, sizeof(buf), "row%zu x=%.2fm", i, r.rows[i].x_m);
    putLabel(img, buf, {img.cols - static_cast<int>(190 * sc), y - 6}, cv::Scalar(255, 255, 255), 0.55 * sc);
  }

  int k0 = 0;  // row 0 라벨을 위아래로 엇갈려서 옆 판 라벨과 안 겹치게
  for (const auto & o : r.obstacles) {
    const cv::Scalar c = (o.color == RED) ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0);
    cv::rectangle(img, o.box, c, o.row == 0 ? 3 : 1);
    // 아랫변: 분홍 = 바닥 접점 (거리 계산에 씀), 주황 = 가려진 끝 (추정)
    const cv::Scalar bc = (o.bottom == OCCLUDED) ? cv::Scalar(0, 128, 255) : cv::Scalar(255, 0, 255);
    cv::line(img, {o.box.x, o.v}, {o.box.x + o.box.width, o.v}, bc, 3);
    cv::circle(img, {o.u, o.v}, 5, cv::Scalar(0, 0, 255), -1);

    const char * name = (o.color == RED) ? "R" : "B";
    if (o.row > 0) {
      std::snprintf(buf, sizeof(buf), "%s%d%s", name, o.row, o.bottom == OCCLUDED ? "*" : "");
      putLabel(img, buf, {o.box.x + 2, o.box.y + static_cast<int>(18 * sc)}, c, 0.5 * sc);
      continue;
    }
    // row 0 은 판 아래 바닥 쪽에 자세히
    const int base = std::min(img.rows - 30, o.v + static_cast<int>((22 + (k0++ % 2) * 40) * sc));
    std::snprintf(buf, sizeof(buf), "%s%s x%.2f y%+.2f", name,
      o.count > 1 ? ("x" + std::to_string(o.count)).c_str() : "", o.x_m, o.y_m);
    putLabel(img, buf, {o.box.x + 2, base}, c, 0.5 * sc);
    std::snprintf(buf, sizeof(buf), "w%.2f [%s]%s%s", o.width_m, bottomTag(o.bottom),
      o.cut_left ? "<" : "", o.cut_right ? ">" : "");
    putLabel(img, buf, {o.box.x + 2, base + static_cast<int>(18 * sc)}, c, 0.5 * sc);
  }

  // 빈 틈: row 0 거리에서 gap 위치를 화면에 찍고 화살표
  // 하늘색 = 빈 틈, 주황 = 전부 막힘 (이쪽으로 옆걸음)
  {
    const cv::Scalar gc = r.gap_found ? cv::Scalar(255, 255, 0) : cv::Scalar(0, 128, 255);
    const double gx = r.detected ? r.nearest_x_m : 1.0;
    cv::Point p;
    if (groundToPixel(gx, r.gap_y_m, cam, p)) {
      p.x = std::clamp(p.x, 0, img.cols - 1);
      p.y = std::clamp(p.y, 0, img.rows - 1);
      cv::arrowedLine(img, {img.cols / 2, img.rows - 5}, p, gc, 4, 8, 0, 0.05);
      cv::circle(img, p, 10, gc, -1);
    }
  }

  // 위쪽 정보 패널
  const int ph = static_cast<int>(70 * sc);
  cv::rectangle(img, {0, 0}, {img.cols, ph}, cv::Scalar(0, 0, 0), -1);
  std::snprintf(buf, sizeof(buf), "obstacles: %zu  rows: %zu  row0 x: %.2fm",
    r.obstacles.size(), r.rows.size(), r.nearest_x_m);
  putLabel(img, buf, {8, static_cast<int>(28 * sc)}, cv::Scalar(255, 255, 255), 0.7 * sc);
  if (r.gap_found) {
    std::snprintf(buf, sizeof(buf), "gap_y: %+.2fm (%s)  width %.2fm", r.gap_y_m,
      std::abs(r.gap_y_m) < 0.05 ? "STRAIGHT" : (r.gap_y_m > 0 ? "LEFT" : "RIGHT"), r.gap_width_m);
  } else {
    std::snprintf(buf, sizeof(buf), "BLOCKED -> step %s (%+.2fm)",
      r.gap_y_m > 0 ? "LEFT" : "RIGHT", r.gap_y_m);
  }
  putLabel(img, buf, {8, static_cast<int>(60 * sc)},
    r.gap_found ? cv::Scalar(255, 255, 0) : cv::Scalar(0, 128, 255), 0.7 * sc);
  return img;
}

cv::Mat drawMask(const Result & r)
{
  cv::Mat img(r.red_mask.size(), CV_8UC3, cv::Scalar(0, 0, 0));
  img.setTo(cv::Scalar(0, 0, 255), r.red_mask);
  img.setTo(cv::Scalar(255, 0, 0), r.blue_mask);
  // 아랫변: 노랑 = 바닥 접점, 주황 = 가려짐, 흰색 = 화면 아래로 잘림
  for (const auto & o : r.obstacles) {
    const cv::Scalar c = o.bottom == GROUND ? cv::Scalar(0, 255, 255) :
      (o.bottom == OCCLUDED ? cv::Scalar(0, 128, 255) : cv::Scalar(255, 255, 255));
    cv::line(img, {o.box.x, o.v}, {o.box.x + o.box.width, o.v}, c, 3);
  }
  return img;
}

}  // namespace obstacle_vision
