#include "task_master/task_master.hpp"
#include "vision_interfaces/msg/obstacle_array.hpp"
#include <chrono>
#include <memory>

namespace task_master
{

TaskMaster::TaskMaster()
: Node("task_master")
{
  const double loop_rate_hz = declare_parameter<double>("loop_rate_hz", 20.0);
  gamecontrol_timeout_sec_ = declare_parameter<double>("gamecontrol_timeout_sec", 2.0);

  const auto gamecontrol_topic =
    declare_parameter<std::string>("topics.gamecontrol_sub", "gamecontroldata");
  const auto imu_topic = declare_parameter<std::string>("topics.imu_sub", "Imu");
  const auto master2ik_topic = declare_parameter<std::string>("topics.master2ik_pub", "master2ik");
  const auto kick_flag_topic = declare_parameter<std::string>("topics.kick_flag_pub", "kick_flag");
  const auto motion_operator_topic =
    declare_parameter<std::string>("topics.motion_operator_pub", "motion_operator");
  const auto motion_end_topic =
    declare_parameter<std::string>("topics.motion_end_sub", "motion_end");

  // ik_walk 쪽 QoS 에 맞춤
  const auto motion_qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable();
  const auto vision_obstacle_topic =
    declare_parameter<std::string>("topics.vision_sub", "vision/obstacles");

  gamecontrol_sub_ = create_subscription<gamecontroller::msg::Gamecontroldata>(
    gamecontrol_topic, 10,
    std::bind(&TaskMaster::gamecontrolCallback, this, std::placeholders::_1));
  // ebimu 가 best_effort 로 publish 하므로 best_effort 로 받는다
  imu_sub_ = create_subscription<humanoid_interfaces::msg::ImuMsg>(
    imu_topic, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
    std::bind(&TaskMaster::imuCallback, this, std::placeholders::_1));
  motion_end_sub_ = create_subscription<humanoid_interfaces::msg::MotionOperator>(
    motion_end_topic, motion_qos,
    std::bind(&TaskMaster::motionEndCallback, this, std::placeholders::_1));
  vision_sub_ = create_subscription<vision_interfaces::msg::ObstacleArray>(
    vision_obstacle_topic, 10,
    std::bind(&TaskMaster::visionCallback, this, std::placeholders::_1));

  master2ik_pub_ = create_publisher<humanoid_interfaces::msg::Master2IkMsg>(master2ik_topic, 10);
  kick_flag_pub_ = create_publisher<std_msgs::msg::String>(kick_flag_topic, 10);
  motion_operator_pub_ =
    create_publisher<humanoid_interfaces::msg::MotionOperator>(motion_operator_topic, motion_qos);

  const auto period = std::chrono::duration<double>(1.0 / loop_rate_hz);
  timer_ = create_wall_timer(
    std::chrono::duration_cast<std::chrono::nanoseconds>(period),
    std::bind(&TaskMaster::tick, this));

  RCLCPP_INFO(get_logger(), "task_master started (%.1f Hz)", loop_rate_hz);
}

// ============================ 콜백 ============================

void TaskMaster::gamecontrolCallback(const gamecontroller::msg::Gamecontroldata::SharedPtr msg)
{
  game_ = *msg;
  last_gamecontrol_time_ = now();
  has_gamecontrol_ = true;
}

void TaskMaster::imuCallback(const humanoid_interfaces::msg::ImuMsg::SharedPtr msg)
{
  imu_ = *msg;
}

void TaskMaster::motionEndCallback(const humanoid_interfaces::msg::MotionOperator::SharedPtr msg)
{
  if (msg->motion_end != 0 && msg->motion_num == running_motion_) {
    RCLCPP_INFO(get_logger(), "motion %d finished", msg->motion_num);
    running_motion_ = -1;
  }
}

void TaskMaster::visionCallback(const vision_interfaces::msg::ObstacleArray::SharedPtr msg)
{
  gap_found_ = msg->gap_found;
  gap_y_m_ = msg->gap_y_m;

  obstacle_data_received_ = true;
}
// ============================ 메인 루프 ============================

void TaskMaster::tick()
{
  // gamecontroller 데이터가 없거나 끊기면 안전하게 INITIAL 로 취급
  GameState state = GameState::INITIAL;
  const bool alive = has_gamecontrol_ &&
    (now() - last_gamecontrol_time_).seconds() < gamecontrol_timeout_sec_;
  if (alive) {
    state = static_cast<GameState>(game_.state);
  }

  state_changed_ = (state != prev_state_);
  if (state_changed_) {
    RCLCPP_INFO(
      get_logger(), "game state: %ld -> %ld",
      static_cast<long>(prev_state_), static_cast<long>(state));
    prev_state_ = state;
  }

  // 모션 실행 중에는 ik_walk 가 모터 publish 를 멈춘 상태이므로 새 명령을 내지 않는다
  if (running_motion_ >= 0) {
    return;
  }

  // PENALTY_NONE = 0
  if (alive && game_.penalty != 0) {
    onPenalized();
    return;
  }

  switch (state) {
    case GameState::INITIAL:  onInitial();  break;
    case GameState::READY:    onReady();    break;
    case GameState::SET:      onSet();      break;
    case GameState::PLAYING:  onPlaying();  break;
    case GameState::FINISHED: onFinished(); break;
    default:                  stopWalk();   break;
  }
}

// ============================ 상태별 처리 ============================
// 기본 동작은 전부 "정지". 경기 로직은 각 함수의 TODO 에 작성.

void TaskMaster::onInitial()
{
  stopWalk();
}

void TaskMaster::onReady()
{
  // TODO: 시작 위치로 이동 (game_.position, game_.myside, game_.iskickoff 참고)
  stopWalk();
}

void TaskMaster::onSet()
{
  // 규정상 SET 에서는 움직이면 안 됨
  stopWalk();
}

void TaskMaster::onPlaying()
{
    if(gap_found_ && gap_y_m_ == 0.0)
    {
      walk();
    }
    else if(gap_found_ && gap_y_m_ < 0)
    {
      leftwalk();
    }
    else if(gap_found_ && gap_y_m_ > 0)
    {
      rightwalk();
    }
  }
  
}

void TaskMaster::onFinished()
{
  stopWalk();
}

void TaskMaster::onPenalized()
{
  stopWalk();
}

// ============================ 명령 헬퍼 ============================

void TaskMaster::walk()
{
  humanoid_interfaces::msg::Master2IkMsg msg;
  msg.x_length = 9;
  msg.y_length = 0;
  msg.yaw = 0;
  msg.flag = 1.0;
  master2ik_pub_->publish(msg);
}

void TaskMaster::leftwalk()
{
  humanoid_interfaces::msg::Master2IkMsg msg;
  msg.x_length = 0;
  msg.y_length = 7;
  msg.yaw = 0;
  msg.flag = 1.0;
  master2ik_pub_->publish(msg);
}

void TaskMaster::rightwalk()
{
  humanoid_interfaces::msg::Master2IkMsg msg;
  // TO DO: 우횡진할 때 값 수정 후 주석 해제
  msg.x_length = 0;
  msg.y_length = -7;
  msg.yaw = 0;
  // msg.flag = 1.0;
  master2ik_pub_->publish(msg);
}

void TaskMaster::stopWalk()
{
  humanoid_interfaces::msg::Master2IkMsg msg;
  msg.flag = 0.0;
  master2ik_pub_->publish(msg);
}

void TaskMaster::kick(const std::string & side)
{
  std_msgs::msg::String msg;
  msg.data = side;
  kick_flag_pub_->publish(msg);
}

void TaskMaster::startMotion(int32_t motion_num)
{
  humanoid_interfaces::msg::MotionOperator msg;
  msg.motion_num = motion_num;
  msg.motion_end = 0;
  motion_operator_pub_->publish(msg);
  running_motion_ = motion_num;
}

}  // namespace task_master

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<task_master::TaskMaster>());
  rclcpp::shutdown();
  return 0;
}
