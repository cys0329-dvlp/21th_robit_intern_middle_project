#ifndef TASK_MASTER__TASK_MASTER_HPP_
#define TASK_MASTER__TASK_MASTER_HPP_

#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include "gamecontroller/msg/gamecontroldata.hpp"
#include "humanoid_interfaces/msg/imu_msg.hpp"
#include "humanoid_interfaces/msg/master2_ik_msg.hpp"
#include "humanoid_interfaces/msg/motion_operator.hpp"
#include "vision_interfaces/msg/obstacle_array.hpp"
#include "vision_interfaces/msg/field_lines.hpp"

namespace task_master
{

// RoboCupGameControlData.h 의 STATE_* 값과 동일
enum class GameState : int64_t
{
  INITIAL = 0,
  READY = 1,
  SET = 2,
  PLAYING = 3,
  FINISHED = 4,
};

class TaskMaster : public rclcpp::Node
{
public:
  TaskMaster();

private:
  // ---------------- 콜백 ----------------
  void gamecontrolCallback(const gamecontroller::msg::Gamecontroldata::SharedPtr msg);
  void imuCallback(const humanoid_interfaces::msg::ImuMsg::SharedPtr msg);
  void motionEndCallback(const humanoid_interfaces::msg::MotionOperator::SharedPtr msg);
  void visionCallback(const vision_interfaces::msg::ObstacleArray::SharedPtr msg);
  void fieldLinesCallback(const vision_interfaces::msg::FieldLines::SharedPtr msg);

  void tick();
  void publishLineStatus();  // 0.1초마다: line_keep_dist_m + 좌우 선까지 거리 -> gamecontroller textEdit

  // ---------------- 상태별 처리 (여기에 경기 로직 작성) ----------------
  void onInitial();
  void onReady();
  void onSet();
  void onPlaying();
  void onFinished();
  void onPenalized();

  // ---------------- 명령 헬퍼 ----------------
  // x, y: 보폭, yaw: 회전. ik_walk 의 master2ik 단위를 그대로 따른다.
  void walk();
  void rightwalk();
  void leftwalk();
  void stopWalk();
  void logDecision(const char * decision);
  void walkKeepInField(const char * straight_label);

  // 실제 보행 방향. 한 번 바꾸면 direction_hold_sec 동안은 다른 방향 요청이 와도 유지
  enum class WalkDir { STRAIGHT, LEFT, RIGHT, STOP };
  void moveDir(WalkDir dir, const char * label);  // 직진 + 경계선에 붙으면 반대쪽으로

  struct WalkCmd
  {
    double x = 0.0;
    double y = 0.0;
    double yaw = 0.0;
  };
  void publishWalk(const WalkCmd & cmd);
  // side: "left" 또는 "right"
  void kick(const std::string & side);
  void startMotion(int32_t motion_num);

  // ---------------- 통신 ----------------
  rclcpp::Subscription<gamecontroller::msg::Gamecontroldata>::SharedPtr gamecontrol_sub_;
  rclcpp::Subscription<humanoid_interfaces::msg::ImuMsg>::SharedPtr imu_sub_;
  rclcpp::Subscription<humanoid_interfaces::msg::MotionOperator>::SharedPtr motion_end_sub_;
  rclcpp::Subscription<vision_interfaces::msg::ObstacleArray>::SharedPtr vision_sub_;
  rclcpp::Subscription<vision_interfaces::msg::FieldLines>::SharedPtr field_lines_sub_;
  rclcpp::Publisher<humanoid_interfaces::msg::Master2IkMsg>::SharedPtr master2ik_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr kick_flag_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr walk_decision_pub_;  // gamecontroller textEdit 로그용
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr line_status_pub_;    // gamecontroller textEdit 상태 줄
  rclcpp::Publisher<humanoid_interfaces::msg::MotionOperator>::SharedPtr motion_operator_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr line_status_timer_;

  // ---------------- 상태 ----------------
  gamecontroller::msg::Gamecontroldata game_;
  humanoid_interfaces::msg::ImuMsg imu_;
  rclcpp::Time last_gamecontrol_time_;
  bool has_gamecontrol_ = false;
  double gamecontrol_timeout_sec_ = 2.0;

  GameState prev_state_ = GameState::INITIAL;
  bool state_changed_ = false;  // 이번 tick 에 상태가 바뀌었으면 true

  int32_t running_motion_ = -1;  // 실행 중인 모션 번호, 없으면 -1

  // 방향별 보행 명령값 (yaml walk.* 로 튜닝)
  WalkCmd straight_cmd_;
  WalkCmd left_cmd_;
  WalkCmd right_cmd_;

  //Vision에서 받은 장애물 통과 정보
  bool gap_found_ = true;
  float gap_y_m_ = 0.0;
  bool detected_ = false;
  float nearest_x_m_ = 0.0;     // row 0 까지 거리 (m)
  double avoid_dist_m_ = 0.90;  // row 0 이 이보다 가까울 때만 gap 명령을 따름
  std::string last_decision_;    // 마지막 보행 판단 (로그용)
  bool obstacle_data_received_ = false;

  //line_vision 에서 받은 경기장 경계선까지 수직 거리 (m). 왼쪽 선은 +, 오른쪽 선은 - 로 옴
  bool left_found = false;
  float left_dist_m = 0.0;
  bool right_found = false;
  float right_dist_m = 0.0;
  double direction_hold_sec_ = 1.0;  // 방향을 바꾸면 이 시간 동안은 무조건 그 방향 (좌우/직진 번갈아 흔들림 방지)
  bool dir_valid_ = false;           // false 면 아직 방향 없음 (PLAYING 진입 직후) -> 첫 요청 바로 따름
  WalkDir cur_dir_ = WalkDir::STOP;  // 지금 실제로 보내는 방향
  std::string cur_label_;            // 그 방향의 로그 이름 (STRAIGHT (far), RIGHT (line) 등)
  rclcpp::Time dir_since_;           // cur_dir_ 로 바꾼 시각
  double line_keep_dist_m_ = 0.3;  // 직진 중 경계선이 이보다 가까우면 반대쪽으로 옆걸음

  // 선이 양쪽 다 사라졌을 때: 사라지기 직전에 보였던 선의 반대쪽으로 line_lost_side_sec 동안 횡진
  bool seen_left_ = false;                 // 선이 보이던 마지막 tick 의 left_found
  bool seen_right_ = false;                // 선이 보이던 마지막 tick 의 right_found
  WalkDir lost_side_dir_ = WalkDir::STRAIGHT;  // 진행 중인 횡진 방향, STRAIGHT 면 없음
  rclcpp::Time lost_since_;                // 선이 사라진 시각
  double line_lost_side_sec_ = 4.0;
  
};

}  // namespace task_master

#endif  // TASK_MASTER__TASK_MASTER_HPP_
