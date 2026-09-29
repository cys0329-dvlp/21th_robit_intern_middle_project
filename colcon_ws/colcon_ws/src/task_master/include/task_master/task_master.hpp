#ifndef TASK_MASTER__TASK_MASTER_HPP_
#define TASK_MASTER__TASK_MASTER_HPP_

#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include "gamecontroller/msg/gamecontroldata.hpp"
#include "humanoid_interfaces/msg/imu_msg.hpp"
#include "humanoid_interfaces/msg/master2_ik_msg.hpp"
#include "humanoid_interfaces/msg/motion_operator.hpp"
// TODO(vision_interfaces): 패키지 만든 후 주석 해제
// #include "vision_interfaces/msg/..."

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
  // TODO(vision_interfaces): vision 콜백 추가
  // void visionCallback(const vision_interfaces::msg::...::SharedPtr msg);

  void tick();

  // ---------------- 상태별 처리 (여기에 경기 로직 작성) ----------------
  void onInitial();
  void onReady();
  void onSet();
  void onPlaying();
  void onFinished();
  void onPenalized();

  // ---------------- 명령 헬퍼 ----------------
  // x, y: 보폭, yaw: 회전. ik_walk 의 master2ik 단위를 그대로 따른다.
  void walk(double x, double y, double yaw);
  void stopWalk();
  // side: "left" 또는 "right"
  void kick(const std::string & side);
  void startMotion(int32_t motion_num);

  // ---------------- 통신 ----------------
  rclcpp::Subscription<gamecontroller::msg::Gamecontroldata>::SharedPtr gamecontrol_sub_;
  rclcpp::Subscription<humanoid_interfaces::msg::ImuMsg>::SharedPtr imu_sub_;
  rclcpp::Subscription<humanoid_interfaces::msg::MotionOperator>::SharedPtr motion_end_sub_;
  rclcpp::Publisher<humanoid_interfaces::msg::Master2IkMsg>::SharedPtr master2ik_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr kick_flag_pub_;
  rclcpp::Publisher<humanoid_interfaces::msg::MotionOperator>::SharedPtr motion_operator_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // ---------------- 상태 ----------------
  gamecontroller::msg::Gamecontroldata game_;
  humanoid_interfaces::msg::ImuMsg imu_;
  rclcpp::Time last_gamecontrol_time_;
  bool has_gamecontrol_ = false;
  double gamecontrol_timeout_sec_ = 2.0;

  GameState prev_state_ = GameState::INITIAL;
  bool state_changed_ = false;  // 이번 tick 에 상태가 바뀌었으면 true

  int32_t running_motion_ = -1;  // 실행 중인 모션 번호, 없으면 -1
};

}  // namespace task_master

#endif  // TASK_MASTER__TASK_MASTER_HPP_
