// 카메라 이미지 -> obstacle_detector 로 처리 -> /vision/obstacles 로 publish
// line_vision 의 /vision/field_lines 가 오고 있으면 경기장 밖은 빈 틈으로 고르지 않음 (안 오면 예전처럼)
// show_window: true 면 imshow 로 결과/마스크 창을 띄우고, 마스크 창의 트랙바로 HSV 를 실시간 튜닝
//   키: s = 현재 프레임 저장, p = 현재 HSV 값 출력 (yaml 에 옮겨 적기), q = 창 닫기

#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>

#include "obstacle_vision/obstacle_detector.hpp"
#include "vision_interfaces/msg/field_lines.hpp"
#include "vision_interfaces/msg/obstacle_array.hpp"

namespace obstacle_vision
{

static const char * RESULT_WIN = "obstacle_vision";
static const char * MASK_WIN = "mask (HSV tuning)";

class ObstacleVisionNode : public rclcpp::Node
{
public:
  ObstacleVisionNode()
  : Node("obstacle_vision")
  {
    // ---------------- 토픽 ----------------
    const auto image_topic =
      declare_parameter<std::string>("image_topic", "/camera1/camera/compressed_image");
    const auto info_topic = declare_parameter<std::string>("camera_info_topic", "/camera1/info");
    const auto out_topic = declare_parameter<std::string>("obstacles_topic", "/vision/obstacles");
    const auto debug_topic =
      declare_parameter<std::string>("debug_image_topic", "/vision/debug_image");
    show_window_ = declare_parameter<bool>("show_window", true);
    display_scale_ = declare_parameter<double>("display_scale", 1.0);

    // ---------------- 카메라 ----------------
    cam_.cam_height = declare_parameter<double>("camera.height", 0.45);
    cam_.tilt_rad = declare_parameter<double>("camera.tilt_deg", 20.0) * M_PI / 180.0;
    // camera_info 가 안 올 때 쓰는 값 (camera_info_config.yaml 의 640x480 캘리브레이션)
    fb_w_ = declare_parameter<double>("camera.fallback.width", 640.0);
    fb_fx_ = declare_parameter<double>("camera.fallback.fx", 471.953641);
    fb_fy_ = declare_parameter<double>("camera.fallback.fy", 476.574144);
    fb_cx_ = declare_parameter<double>("camera.fallback.cx", 309.509126);
    fb_cy_ = declare_parameter<double>("camera.fallback.cy", 228.222101);

    // ---------------- 색 ----------------
    cp_.red_h_low = declare_parameter<int>("color.red_h_low", cp_.red_h_low);
    cp_.red_h_high = declare_parameter<int>("color.red_h_high", cp_.red_h_high);
    cp_.red_s_min = declare_parameter<int>("color.red_s_min", cp_.red_s_min);
    cp_.red_v_min = declare_parameter<int>("color.red_v_min", cp_.red_v_min);
    cp_.blue_h_min = declare_parameter<int>("color.blue_h_min", cp_.blue_h_min);
    cp_.blue_h_max = declare_parameter<int>("color.blue_h_max", cp_.blue_h_max);
    cp_.blue_s_min = declare_parameter<int>("color.blue_s_min", cp_.blue_s_min);
    cp_.blue_v_min = declare_parameter<int>("color.blue_v_min", cp_.blue_v_min);
    cp_.morph_kernel = declare_parameter<int>("color.morph_kernel", cp_.morph_kernel);
    cp_.min_area_ratio = declare_parameter<double>("color.min_area_ratio", cp_.min_area_ratio);
    cp_.split_jump_ratio =
      declare_parameter<double>("color.split_jump_ratio", cp_.split_jump_ratio);

    // ---------------- 빈 틈 ----------------
    gp_.obstacle_width = declare_parameter<double>("gap.obstacle_width", gp_.obstacle_width);
    gp_.row_tol_ratio = declare_parameter<double>("gap.row_tol_ratio", gp_.row_tol_ratio);
    gp_.robot_half_width = declare_parameter<double>("gap.robot_half_width", gp_.robot_half_width);
    gp_.search_half_width =
      declare_parameter<double>("gap.search_half_width", gp_.search_half_width);
    gp_.max_range = declare_parameter<double>("gap.max_range", gp_.max_range);
    gp_.min_height = declare_parameter<double>("gap.min_height", gp_.min_height);

    // ---------------- 경기장 경계선 (line_vision) ----------------
    use_field_ = declare_parameter<bool>("field.use_lines", true);
    const auto field_topic =
      declare_parameter<std::string>("field.topic", "/vision/field_lines");
    field_timeout_ = declare_parameter<double>("field.timeout_s", 0.5);
    field_max_angle_ = declare_parameter<double>("field.max_line_angle_deg", 60.0) * M_PI / 180.0;
    field_width_ = declare_parameter<double>("field.width", 1.4);
    field_margin_ = declare_parameter<double>("field.margin", 0.20);

    // ---------------- 통신 ----------------
    obstacles_pub_ = create_publisher<vision_interfaces::msg::ObstacleArray>(out_topic, 10);
    debug_pub_ = create_publisher<sensor_msgs::msg::Image>(debug_topic, 1);
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      info_topic, 10, std::bind(&ObstacleVisionNode::infoCallback, this, std::placeholders::_1));
    if (use_field_) {
      field_sub_ = create_subscription<vision_interfaces::msg::FieldLines>(
        field_topic, 10, [this](const vision_interfaces::msg::FieldLines::SharedPtr msg) {
          field_msg_ = *msg;
          field_time_ = std::chrono::steady_clock::now();
          has_field_ = true;
        });
    }

    // image_path 가 있으면 카메라 대신 사진으로 테스트
    const auto image_path = declare_parameter<std::string>("image_path", "");
    if (!image_path.empty()) {
      test_image_ = cv::imread(image_path);
      if (test_image_.empty()) {
        RCLCPP_ERROR(get_logger(), "사진을 못 읽음: %s", image_path.c_str());
      } else {
        RCLCPP_INFO(get_logger(), "사진 테스트 모드: %s", image_path.c_str());
        timer_ = create_wall_timer(std::chrono::milliseconds(100), [this]() {
            std_msgs::msg::Header h;
            h.stamp = now();
            h.frame_id = "camera";
            process(test_image_, h);
          });
      }
    } else {
      image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic, rclcpp::SensorDataQoS(),
        std::bind(&ObstacleVisionNode::imageCallback, this, std::placeholders::_1));
    }

    if (show_window_) {
      setupWindows();
    }
    RCLCPP_INFO(get_logger(), "obstacle_vision started. image: %s -> %s",
      image_topic.c_str(), out_topic.c_str());
  }

  ~ObstacleVisionNode() override
  {
    if (show_window_) {
      cv::destroyAllWindows();
    }
  }

private:
  void infoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    info_ = *msg;
    has_info_ = true;
  }

  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv::Mat frame;
    try {
      frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
    } catch (cv_bridge::Exception & e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge: %s", e.what());
      return;
    }
    process(frame, msg->header);
  }

  // 최근(field.timeout_s 안) 받은 경계선 -> FieldInfo. 오래됐거나 없으면 valid=false (경기장 제한 안 함)
  FieldInfo makeFieldInfo() const
  {
    FieldInfo f;
    f.width = field_width_;
    f.margin = field_margin_;
    if (!has_field_) {return f;}
    const double age = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - field_time_).count();
    if (age > field_timeout_) {return f;}

    const auto & m = field_msg_;
    auto set = [this](FieldLine & l, bool found, double dist, double angle_deg) {
        const double a = angle_deg * M_PI / 180.0;
        l.found = found && std::abs(a) < field_max_angle_;  // 거의 옆으로 누운 선은 안 믿음
        l.dist = dist;
        l.angle = a;
      };
    set(f.left, m.left_found, m.left_dist_m, m.left_angle_deg);
    set(f.right, m.right_found, m.right_dist_m, m.right_angle_deg);
    f.valid = f.left.found || f.right.found;
    return f;
  }

  // 이미지 해상도에 맞게 fx, fy, cx, cy 를 맞춘다 (camera_info 해상도와 다를 수 있음)
  void updateIntrinsics(const cv::Mat & frame)
  {
    if (has_info_ && info_.width > 0) {
      const double s = static_cast<double>(frame.cols) / info_.width;
      cam_.fx = info_.k[0] * s;
      cam_.cx = info_.k[2] * s;
      cam_.fy = info_.k[4] * s;
      cam_.cy = info_.k[5] * s;
    } else {
      const double s = frame.cols / fb_w_;
      cam_.fx = fb_fx_ * s;
      cam_.cx = fb_cx_ * s;
      cam_.fy = fb_fy_ * s;
      cam_.cy = fb_cy_ * s;
    }
  }

  void process(const cv::Mat & frame, const std_msgs::msg::Header & header)
  {
    const auto t0 = std::chrono::steady_clock::now();
    if (show_window_) {
      readTrackbars();
    }
    updateIntrinsics(frame);

    const Result r = detect(frame, cp_, cam_, gp_, makeFieldInfo());
    publish(r, header);

    const double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();

    cv::Mat vis = drawResult(frame, r, cam_);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f ms  %s", ms, has_info_ ? "info:OK" : "info:fallback");
    cv::putText(vis, buf, {8, vis.rows - 10}, cv::FONT_HERSHEY_SIMPLEX, 0.6,
      cv::Scalar(255, 255, 255), 2);

    if (debug_pub_->get_subscription_count() > 0) {
      debug_pub_->publish(*cv_bridge::CvImage(header, "bgr8", vis).toImageMsg());
    }

    if (show_window_) {
      showWindows(frame, vis, drawMask(r));
    }
  }

  void publish(const Result & r, const std_msgs::msg::Header & header)
  {
    vision_interfaces::msg::ObstacleArray msg;
    msg.header = header;
    msg.detected = r.detected;
    msg.nearest_x_m = r.nearest_x_m;
    msg.gap_found = r.gap_found;
    msg.gap_y_m = r.gap_y_m;
    msg.gap_width_m = r.gap_width_m;
    for (const auto & row : r.rows) {
      msg.row_x_m.push_back(row.x_m);
    }
    for (const auto & o : r.obstacles) {
      vision_interfaces::msg::Obstacle m;
      m.color = (o.color == RED) ? m.RED : m.BLUE;
      m.x_m = o.x_m;
      m.y_m = o.y_m;
      m.width_m = o.width_m;
      m.u = o.u;
      m.v = o.v;
      m.w_px = o.w_px;
      m.count = o.count;
      m.row = o.row;
      m.bottom = static_cast<uint8_t>(o.bottom);
      m.cut_left = o.cut_left;
      m.cut_right = o.cut_right;
      msg.obstacles.push_back(m);
    }
    obstacles_pub_->publish(msg);
  }

  // ---------------- imshow ----------------

  void setupWindows()
  {
    cv::namedWindow(RESULT_WIN, cv::WINDOW_NORMAL);
    cv::namedWindow(MASK_WIN, cv::WINDOW_NORMAL);
    cv::createTrackbar("red H low max", MASK_WIN, nullptr, 180);
    cv::createTrackbar("red H high min", MASK_WIN, nullptr, 180);
    cv::createTrackbar("red S min", MASK_WIN, nullptr, 255);
    cv::createTrackbar("red V min", MASK_WIN, nullptr, 255);
    cv::createTrackbar("blue H min", MASK_WIN, nullptr, 180);
    cv::createTrackbar("blue H max", MASK_WIN, nullptr, 180);
    cv::createTrackbar("blue S min", MASK_WIN, nullptr, 255);
    cv::createTrackbar("blue V min", MASK_WIN, nullptr, 255);
    cv::setTrackbarPos("red H low max", MASK_WIN, cp_.red_h_low);
    cv::setTrackbarPos("red H high min", MASK_WIN, cp_.red_h_high);
    cv::setTrackbarPos("red S min", MASK_WIN, cp_.red_s_min);
    cv::setTrackbarPos("red V min", MASK_WIN, cp_.red_v_min);
    cv::setTrackbarPos("blue H min", MASK_WIN, cp_.blue_h_min);
    cv::setTrackbarPos("blue H max", MASK_WIN, cp_.blue_h_max);
    cv::setTrackbarPos("blue S min", MASK_WIN, cp_.blue_s_min);
    cv::setTrackbarPos("blue V min", MASK_WIN, cp_.blue_v_min);
  }

  void readTrackbars()
  {
    cp_.red_h_low = cv::getTrackbarPos("red H low max", MASK_WIN);
    cp_.red_h_high = cv::getTrackbarPos("red H high min", MASK_WIN);
    cp_.red_s_min = cv::getTrackbarPos("red S min", MASK_WIN);
    cp_.red_v_min = cv::getTrackbarPos("red V min", MASK_WIN);
    cp_.blue_h_min = cv::getTrackbarPos("blue H min", MASK_WIN);
    cp_.blue_h_max = cv::getTrackbarPos("blue H max", MASK_WIN);
    cp_.blue_s_min = cv::getTrackbarPos("blue S min", MASK_WIN);
    cp_.blue_v_min = cv::getTrackbarPos("blue V min", MASK_WIN);
  }

  void showWindows(const cv::Mat & frame, const cv::Mat & vis, const cv::Mat & mask)
  {
    if (display_scale_ != 1.0) {
      cv::Mat a, b;
      cv::resize(vis, a, {}, display_scale_, display_scale_);
      cv::resize(mask, b, {}, display_scale_, display_scale_);
      cv::imshow(RESULT_WIN, a);
      cv::imshow(MASK_WIN, b);
    } else {
      cv::imshow(RESULT_WIN, vis);
      cv::imshow(MASK_WIN, mask);
    }

    const int key = cv::waitKey(1) & 0xFF;
    if (key == 's') {
      const std::string path = "/tmp/obstacle_" + std::to_string(save_count_++) + ".png";
      cv::imwrite(path, frame);
      RCLCPP_INFO(get_logger(), "saved %s", path.c_str());
    } else if (key == 'p') {
      RCLCPP_INFO(get_logger(),
        "\ncolor:\n  red_h_low: %d\n  red_h_high: %d\n  red_s_min: %d\n  red_v_min: %d\n"
        "  blue_h_min: %d\n  blue_h_max: %d\n  blue_s_min: %d\n  blue_v_min: %d",
        cp_.red_h_low, cp_.red_h_high, cp_.red_s_min, cp_.red_v_min,
        cp_.blue_h_min, cp_.blue_h_max, cp_.blue_s_min, cp_.blue_v_min);
    } else if (key == 'q') {
      show_window_ = false;
      cv::destroyAllWindows();
    }
  }

  // ---------------- 멤버 ----------------
  ColorParams cp_;
  CameraModel cam_;
  GapParams gp_;
  double fb_w_, fb_fx_, fb_fy_, fb_cx_, fb_cy_;

  bool use_field_ = true;
  double field_timeout_, field_max_angle_, field_width_, field_margin_;
  vision_interfaces::msg::FieldLines field_msg_;
  std::chrono::steady_clock::time_point field_time_;
  bool has_field_ = false;

  sensor_msgs::msg::CameraInfo info_;
  bool has_info_ = false;
  bool show_window_ = true;
  double display_scale_ = 1.0;
  int save_count_ = 0;
  cv::Mat test_image_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Subscription<vision_interfaces::msg::FieldLines>::SharedPtr field_sub_;
  rclcpp::Publisher<vision_interfaces::msg::ObstacleArray>::SharedPtr obstacles_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace obstacle_vision

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<obstacle_vision::ObstacleVisionNode>());
  rclcpp::shutdown();
  return 0;
}
