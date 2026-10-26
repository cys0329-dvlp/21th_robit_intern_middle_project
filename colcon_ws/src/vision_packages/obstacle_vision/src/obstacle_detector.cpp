#include "obstacle_vision/obstacle_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace obstacle_vision
{

// 색 마스크

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

  // 트랙바로 빨강, 파랑 범위가 겹치게 잡히면 한 픽셀이 두 색이 되니 파랑 우선으로 정리
  red.setTo(0, blue);

  // open, close
  const int k = std::max(1, p.morph_kernel);
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
  for (cv::Mat * m : {&red, &blue}) {
    cv::morphologyEx(*m, *m, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(*m, *m, cv::MORPH_CLOSE, kernel);
  }
}

// 마스크에서 조각으로

namespace
{

struct Run
{
  int x, top, bot;
  Bottom st;
};

// y = a*x + b  뒤 robustline을 위한 line 구조체
struct Line
{
  double a = 0, b = 0;
  int inliers = 0;
  double at(double x) const {return a * x + b;}
};

// 판의 아랫변과 윗변은 거의 수평일거라는 전제로 함
// 이보다 가파르면 판 옆 모서리 왜냐면 카메라를 위에서 아래로 보면 기울여져 보이니까
constexpr double MAX_EDGE_SLOPE = 0.35;

Line robustLine(const std::vector<int> & xs, const std::vector<int> & ys, double thr)
{
  Line l;
  std::vector<int> tmp(ys);
  std::nth_element(tmp.begin(), tmp.begin() + tmp.size() / 2, tmp.end());
  l.b = tmp[tmp.size() / 2];

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

// 판이면 true, 아니면 false o 에 판 정보 채움
bool makeObstacle(const std::vector<Run> & runs, const SegCtx & c, Obstacle & o)
{
  std::vector<int> xs, bots;
  for (const auto & r : runs) {
    xs.push_back(r.x);
    bots.push_back(r.bot);
  }
  // 아랫변은 직선으로 맞춤 
  const double thr = std::max(2, c.jump / 2);
  const Line bl = robustLine(xs, bots, thr);
  if (std::abs(bl.a) > MAX_EDGE_SLOPE || bl.inliers < c.min_w) {return false;}

  //판의 좌우범위 정하기
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
  o.v = static_cast<int>(std::lround(bl.at(o.u)));  //(u,v)는 결국 아랫변의 중앙점 이걸로 나중에 판 까지와의 거리를 계산
  std::nth_element(tops.begin(), tops.begin() + tops.size() / 2, tops.end());
  const int top = std::min(tops[tops.size() / 2], o.v); //윗변의 중앙값들
  o.box = cv::Rect(x0, top, x1 - x0 + 1, std::max(1, o.v - top + 1)); //박스 그리기
  o.w_px = x1 - x0 + 1; //픽셀 폭 거리 정보와 곱해서 실제 m로 환산하기 위해 빼둠

  // 아랫변 상태는 다수결로 결정 (그라운드, 오큘드, 컷)
  int cnt[3] = {0, 0, 0};
  for (const auto & r : runs) {cnt[r.st]++;}
  o.bottom = static_cast<Bottom>(std::max_element(cnt, cnt + 3) - cnt);

  o.cut_left = x0 <= c.margin;
  o.cut_right = x1 >= c.W - 1 - c.margin;
  return true;
}

void processSegment(  //뒤판 때어내기
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
 



  const bool top_cut = top_line.at((xs.front() + xs.back()) / 2.0) <= c.margin;
  const bool top_ok = std::abs(top_line.a) <= MAX_EDGE_SLOPE &&
    top_line.inliers * 2 >= static_cast<int>(xs.size());
  if (!top_cut && top_ok && depth < 2) {
    
    std::vector<Run> up;
    auto flush = [&]() {
        if (!up.empty() && up.back().x - up.front().x + 1 >= c.min_w) {
          processSegment(up, c, depth + 1, out);
        }   //람다를 이용해서 flush만들기   processSegment 호출
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

}  // namespace 끝

std::vector<Obstacle> findSegments(
  const cv::Mat & red, const cv::Mat & blue, const ColorParams & p, int edge_margin_px)
{
  std::vector<Obstacle> out;
  const int H = red.rows;
  const int W = red.cols;
  const int jump = std::max(2, static_cast<int>(p.split_jump_ratio * H));
  const int look = std::max(3, jump / 2);  // 아랫변 바로 아래 이 픽셀 안에 판이 있으면 가려진 것

  
  cv::Mat rt = red.t();
  cv::Mat bt = blue.t();

  for (Color color : {RED, BLUE}) {
    const cv::Mat & mt = (color == RED) ? rt : bt;

    
    std::vector<std::vector<Run>> cols(W);    //벡터안에 벡터는 2차원 나타내는거임
    for (int x = 0; x < W; x++) {
      const uchar * m = mt.ptr<uchar>(x);
      const uchar * ra = rt.ptr<uchar>(x);
      const uchar * ba = bt.ptr<uchar>(x);
      int y = 0;
      while (y < H) {
        if (!m[y]) {y++; continue;} //색이 없으면 그냥 내려가기
        const int top = y;
        while (y < H && m[y]) {y++;}    //색이 있으면 그 y를 top으로 기억하고 색이 끝날때까지 내려가기
        const int bot = y - 1;          // 끝나기 직전 y가 bot
        if (bot - top + 1 < 2) {continue;}      //2px미만 무시하기

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
        for (int back = 1; back <= 2 && best_i < 0; back++) {
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


    SegCtx c{color, W, H, jump,
      std::max(2, static_cast<int>(p.min_seg_width_ratio * W)), edge_margin_px,
      p.min_area_ratio * W * H};
    for (const auto & ch : chains) {
      processSegment(ch, c, 0, out);
    }
  }
  return out;
}

// px <-> m 변환

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

// 행 나누기 
void assignRows(Result & r, const GapParams & g)
{
  std::vector<Row> & rows = r.rows;
  const double tol = g.row_tol_ratio;

  
  std::vector<Obstacle *> vis, occ;
  for (auto & o : r.obstacles) {(o.bottom == OCCLUDED ? occ : vis).push_back(&o);}
  std::sort(vis.begin(), vis.end(), [](auto * a, auto * b) {return a->x_m < b->x_m;});
  std::vector<std::vector<double>> members;
  double row_start = 0;
  for (auto * o : vis) {    //가까운 순으로 보면서 지금 행의 25퍼센트 이상 멀면 새 행으로 나눔
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

  //가려진 판의 실제 거리는 측정할 수 없지만 범위는 알 수 있다. 가린 판x_lo 보다는 멀고 끝에 보이는 판x_hi 보다는 가깝다고 판단 가능 

  std::sort(occ.begin(), occ.end(), [](auto * a, auto * b) {return a->x_m < b->x_m;});
  for (auto * o : occ) {
    const double x_hi = o->x_m; 
    const double x_lo = o->occluder_x_m;  
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

  //행 번호를 가까운 순으로 정렬
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


//row0 판이 화면에 아래로 잘려서 정확한 거리를 잴 수 없을떄 사용하는 함수. 이때쯤이면 row1 판은 보일거라고 생각하고 row1판으로 row0의 거리를 계산한다. row0 거리 = row1거리 - 1m 로 한다. 경기장의 규칙성을 이용
void estimateCutRow0(Result & r, const GapParams & g, const CameraModel & cam)
{
  if (r.rows.size() < 2 || g.row_spacing <= 0) {return;}  //물론 row1이 없으면 안됨
  bool all_cut = true, row1_ground = false;
  for (const auto & o : r.obstacles) {
    if (o.row == 0 && o.bottom != CUT) {all_cut = false;}
    if (o.row == 1 && o.bottom == GROUND) {row1_ground = true;}
  }
  if (!all_cut || !row1_ground) {return;}
  const double est = r.rows[1].x_m - g.row_spacing;
  if (est < 0.1 || est >= r.rows[0].x_m) {return;}
  r.rows[0].x_m = est;  //row0
  for (auto & o : r.obstacles) {
    if (o.row == 0 && o.x_m > est) {setAtDistance(o, est, cam);}
  }
}



// 같은 행 조각 합치기
//필요한 이유 1) 앞 판에 의해 뒷 판이 가운데만 가려진다. 이 경우 그라운드-오큘드-그라운드의 형턔로 3개로 갈라질 수 있다. 
// 2) 빨강-빨강-빈칸 과 같은 상황이 있다고 보자. 두 색갈 칸의 간격은 0.07미터 정도로 추산 가능한데 이 경우 로봇이 진입하지 못하므로 차라리 한 개의 판으로 보는게 정확할 것이다.
// 3) 판 아랫변에 홈이나 빨강색 같은 이물질이 있을 수 있다. 이경우 판이 몇개로 보일지 모르므로 조취를 취해야 한다.

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

// 한쪽 옆이 앞 행 판이나 화면 끝에 가려 폭이 판 1개보다 좁게 보이면, 보이는 쪽 모서리는 그대로 두고 가려진 쪽으로 판 1개 폭까지 늘린다 (보이는 부분 가운데를 판 가운데로 보내면 위치가 틀어지므로)

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

// 빈틈
// 장애물만 보고 어디로 가라고 명령하면 경기장 선을 넘어선 곳도 경로에 추가되는 문제가 있다. 이를 위해 무조건  경기장 안 쪽으로 가게 하기 위해서는 경기장의 안 쪽 범우 구하기가 필요하다.
// 앞 x(m) 에서 경기장 안쪽 범위 -> r.field_lo / field_hi (로봇 중심이 갈 수 있는 곳)
// 선이 하나만 보이면 반대쪽 선은 경기장 폭만큼 떨어져 있다고 본다
void fieldRange(Result & r, const FieldInfo & f, double x)
{
  r.field = f;
  if (!f.valid) {return;}
  double yl, yr;
  if (f.left.found && f.right.found) {
    yl = f.left.yAt(x);
    yr = f.right.yAt(x);
  } else if (f.left.found) {
    yl = f.left.yAt(x);
    yr = yl - f.width / std::cos(f.left.angle);
  } else if (f.right.found) {
    yr = f.right.yAt(x);
    yl = yr + f.width / std::cos(f.right.angle);
  } else {
    return;
  }
  if (yl - yr < 2 * f.margin + 0.1) {return;}  // 선이 뒤바뀌었거나 너무 좁음 -> 리턴
  r.field_used = true;
  r.field_lo = yr + f.margin;
  r.field_hi = yl - f.margin;
}

void findGap(Result & r, const CameraModel & cam, const GapParams & g, const FieldInfo & f, int img_w)
{
  r.detected = !r.obstacles.empty();
  if (r.detected) {r.nearest_x_m = r.rows[0].x_m;}
  fieldRange(r, f, r.detected ? r.nearest_x_m : 1.0);

  if (!r.detected) {
    // 앞에 아무것도 없음 -> 그대로 직진 (경계선에 붙어 있으면 안쪽으로)
    r.gap_found = true;
    r.gap_y_m = 0;
    r.gap_width_m = 2 * g.search_half_width;
    if (r.field_used) {
      r.gap_y_m = std::clamp(0.0, r.field_lo, r.field_hi);
      r.gap_width_m = r.field_hi - r.field_lo;
    }
    return;
  }

  // row 0 장애물이 막고 있는 좌우 구간 (로봇 반폭 half_w 만큼 넓혀서).
  // 화면 끝에 걸린 판은 화면 밖으로 계속 이어진다고 본다
  constexpr double INF = 1e3;
  auto blockedFor = [&](double half_w) {
      std::vector<std::pair<double, double>> blocked;
      for (const auto & o : r.obstacles) {
        if (o.row != 0) {continue;}
        const double half = o.width_m / 2 + half_w;
        double lo = o.y_m - half, hi = o.y_m + half;
        if (o.bottom == CUT) {
          // 아랫변이 화면 밑으로 잘림 -> 진짜 거리는 x_m 보다 가까울 수 있음 (near_x_m 까지).
          // 가까울수록 같은 픽셀이 정면 쪽에 있으므로 near_x_m 에서의 좌우 끝도 막힌 걸로 친다
          const double zn = depthAt(std::min(g.near_x_m, o.x_m), cam);
          const double yl = -zn * (o.box.x - cam.cx) / cam.fx;
          const double yr = -zn * (o.box.x + o.box.width - cam.cx) / cam.fx;
          lo = std::min(lo, yr - half_w);
          hi = std::max(hi, yl + half_w);
        }
        if (o.cut_left) {hi = INF;}    // 화면 왼쪽 = +y
        if (o.cut_right) {lo = -INF;}
        blocked.push_back({lo, hi});
      }
      std::sort(blocked.begin(), blocked.end());
      return blocked;
    };

  // 화면 밖은 모르는 곳이라 빈 곳으로 치지 않는다
  const double half_fov_tan = (img_w / 2.0) / cam.fx;
  r.view_half_width = std::min(g.search_half_width, depthAt(r.nearest_x_m, cam) * half_fov_tan);
  double lo = -r.view_half_width;
  double hi = r.view_half_width;
  // 경기장 밖도 빈 곳으로 치지 않는다
  if (r.field_used) {
    lo = std::max(lo, r.field_lo);
    hi = std::min(hi, r.field_hi);
    if (lo >= hi) {
      // 보이는 범위가 전부 경기장 밖 -> 경기장 안쪽으로 옆걸음
      r.gap_found = false;
      r.gap_y_m = std::clamp(0.0, r.field_lo, r.field_hi);
      r.gap_width_m = 0;
      return;
    }
  }

  // 막힌 구간 합치기 -> 사이사이가 빈 구간
  auto freeOf = [&](const std::vector<std::pair<double, double>> & blocked) {
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
      return free_iv;
    };
  const auto blocked = blockedFor(g.robot_half_width);
  auto free_iv = freeOf(blocked);
  auto hasFree = [](const auto & iv) {
      return std::any_of(iv.begin(), iv.end(), [](const auto & f) {return f.second > f.first;});
    };
  if (!hasFree(free_iv) && g.robot_min_half_width < g.robot_half_width) {
    // 여유까지 넣으면 안 맞지만 로봇 몸만은 들어가는 틈 (판 사이를 지나는 중일 때 등)
    // 멈추면 판 사이에 끼므로 그 틈 가운데로 간다 (가운데라 양쪽 여유가 제일 큼)
    free_iv = freeOf(blockedFor(g.robot_min_half_width));
  }

  // 보이는 범위가 전부 막혔으면 막힌 덩어리의 좌우 끝 중 가까운 쪽을 가리킨다
  // (한 행에 최대 2개라 옆걸음하면 빈 칸이 보이기 시작함). 경기장 밖으로 나가는 끝은 안 고름
  const double edge_l = std::max_element(blocked.begin(), blocked.end(),
      [](const auto & a, const auto & b) {return a.second < b.second;})->second;
  const double edge_r = blocked.front().first;
  const bool l_ok = edge_l < INF / 2 && (!r.field_used || edge_l <= r.field_hi);
  const bool r_ok = edge_r > -INF / 2 && (!r.field_used || edge_r >= r.field_lo);
  if (l_ok && (!r_ok || std::abs(edge_l) < std::abs(edge_r))) {
    r.gap_y_m = edge_l;
  } else if (r_ok) {
    r.gap_y_m = edge_r;
  } else if (r.field_used && edge_l < INF / 2 && edge_r > -INF / 2) {
    // 양쪽 끝 다 경기장 밖 -> 덜 삐져나가는 쪽 (그쪽이 판과 선 사이가 더 넓음), 경기장 안까지만
    r.gap_y_m = (edge_l - r.field_hi < r.field_lo - edge_r) ? r.field_hi : r.field_lo;
  } else {
    r.gap_y_m = 0;  // 양쪽 다 화면 밖까지 막힘 -> 한 발 물러서서 다시 봐야 함
  }
  r.gap_width_m = 0;

  // 빈 구간 중 정면(0)에서 제일 가까운 구간 (옆걸음이 제일 적게 드는 곳) 을 고르고, 목표는 그 구간의 가운데 (끝에 붙으면 걷다 흔들려서 판에 박음)
  
  r.gap_found = false;
  double best = 1e9;
  for (const auto & f : free_iv) {
    if (f.second <= f.first) {continue;}
    const double d = std::abs(std::clamp(0.0, f.first, f.second));
    if (d < best) {
      best = d;
      r.gap_found = true;
      r.gap_y_m = (f.first + f.second) / 2;
      r.gap_width_m = f.second - f.first;
    }
  }
}

}  // namespace 끝

// 전체

Result detect(
  const cv::Mat & bgr, const ColorParams & cp, const CameraModel & cam, const GapParams & g,
  const FieldInfo & field)
{
  Result r;
  const int H = bgr.rows;
  const int W = bgr.cols;
  makeMasks(bgr, cp, r.red_mask, r.blue_mask);
  std::vector<Obstacle> segs = findSegments(r.red_mask, r.blue_mask, cp, g.edge_margin_px);

  // 가려진 조각 바로 아래에서 가리고 있는 판의 아랫변 찾기 (가로로 가장 많이 겹치는 것)
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

  // 아랫변 바로 아래에서 양옆이 같은 색이면 바닥이 아니라 앞 판에 난 구멍 
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

  // 픽셀 -> 거리(m) 변환, 보이는 높이(m) 계산, 행 나누기
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
    estimateCutRow0(r, g, cam);
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

  findGap(r, cam, g, field, W);
  return r;
}

// 디버그 그리기

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

  // line_vision 경계선 (보라) + 빈 틈 찾을 때 쓴 경기장 안쪽 끝 (row 0 거리에 보라 점)
  if (r.field.valid) {
    for (const FieldLine * l : {&r.field.left, &r.field.right}) {
      if (!l->found) {continue;}
      cv::Point a, b;
      bool have_a = false;
      for (double x = 0.3; x < 4.0; x += 0.1) {
        if (!groundToPixel(x, l->yAt(x), cam, b)) {continue;}
        if (have_a) {cv::line(img, a, b, cv::Scalar(255, 0, 200), 2);}
        a = b;
        have_a = true;
      }
    }
  }
  if (r.field_used) {
    const double fx = r.detected ? r.nearest_x_m : 1.0;
    for (double y : {r.field_lo, r.field_hi}) {
      cv::Point p;
      if (groundToPixel(fx, y, cam, p)) {cv::circle(img, p, 7, cv::Scalar(255, 0, 200), -1);}
    }
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
  const int ph = static_cast<int>(98 * sc);
  cv::rectangle(img, {0, 0}, {img.cols, ph}, cv::Scalar(0, 0, 0), -1);
  std::snprintf(buf, sizeof(buf), "obstacles: %zu  rows: %zu  row0 x: %.2fm",
    r.obstacles.size(), r.rows.size(), r.nearest_x_m);
  putLabel(img, buf, {8, static_cast<int>(28 * sc)}, cv::Scalar(255, 255, 255), 0.7 * sc);
  if (r.gap_found) {
    std::snprintf(buf, sizeof(buf), "gap_y: %+.2fm (%s)  width %.2fm", r.gap_y_m,
      std::abs(r.gap_y_m) < 0.05 ? "STRAIGHT" : (r.gap_y_m > 0 ? "LEFT" : "RIGHT"), r.gap_width_m);
  } else {
    std::snprintf(buf, sizeof(buf), "BLOCKED -> %s (%+.2fm)",
      r.gap_y_m == 0 ? "STOP" : (r.gap_y_m > 0 ? "step LEFT" : "step RIGHT"), r.gap_y_m);
  }
  putLabel(img, buf, {8, static_cast<int>(60 * sc)},
    r.gap_found ? cv::Scalar(255, 255, 0) : cv::Scalar(0, 128, 255), 0.7 * sc);
  if (r.field_used) {
    std::snprintf(buf, sizeof(buf), "field: %s%s  in [%+.2f, %+.2f]m",
      r.field.left.found ? "L" : "-", r.field.right.found ? "R" : "-", r.field_lo, r.field_hi);
  } else {
    std::snprintf(buf, sizeof(buf), "field: no lines (search +-%.1fm)", r.view_half_width);
  }
  putLabel(img, buf, {8, static_cast<int>(90 * sc)}, cv::Scalar(255, 0, 200), 0.6 * sc);
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
