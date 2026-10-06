#include <cmath>
#include "task_master/task_master.hpp"
#include "vision_interfaces/msg/obstacle_array.hpp"
#include <chrono>
#include <cstdlib>
#include <memory>

namespace task_master
{

TaskMaster::TaskMaster()
: Node("task_master")
{
  const double loop_rate_hz = declare_parameter<double>("loop_rate_hz", 20.0);
  gamecontrol_timeout_sec_ = declare_parameter<double>("gamecontrol_timeout_sec", 2.0);

  // 방향별 보행 명령값 (하드웨어 좌우 편차 보정용, ik_walk 의 Tuning_Side 가 추가로 더해짐)
  straight_cmd_ = {
    declare_parameter<double>("walk.straight.x", 12.5),
    declare_parameter<double>("walk.straight.y", 0.0),
    declare_parameter<double>("walk.straight.yaw", 0.0)};
  left_cmd_ = {
    declare_parameter<double>("walk.left.x", 1.5),
    declare_parameter<double>("walk.left.y", 9.5),
    declare_parameter<double>("walk.left.yaw", 0.0)};
  right_cmd_ = {
    declare_parameter<double>("walk.right.x", 0.0),
    declare_parameter<double>("walk.right.y", -6.5),
    declare_parameter<double>("walk.right.yaw", 0.0)};
  avoid_dist_m_ = declare_parameter<double>("avoid_dist_m", 0.90);

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
  detected_ = msg->detected;
  nearest_x_m_ = msg->nearest_x_m;

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
  // 이미 떠 있으면 다시 띄우지 않음 (중복 실행 시 시리얼 포트 충돌 -> 통신 에러)
  if(state_changed_ && std::system("pgrep -f '[d]ynamixel_hardware_interface_node' > /dev/null") != 0)
  {
    std::system("nohup bash -c "
      "'source /home/robit/Desktop/1/task_master/colcon_ws/install/setup.bash && "
      "ros2 launch dynamixel_hardware_interface dynamixel_hardware.launch.py' "
      "> /tmp/dynamixel.log 2>&1 &"
    );
  }

  stopWalk();
}

void TaskMaster::onSet()
{
  // 이미 떠 있으면 다시 띄우지 않음 (ik_walk 가 두 개면 서로 다른 명령을 보냄)
  if (state_changed_ && std::system("pgrep -x ik_walk > /dev/null") != 0) {
    std::system(
      "nohup bash -c "
      "'source /home/robit/Desktop/1/task_master/colcon_ws/install/setup.bash && "
      "ros2 run ik_walk ik_walk' "
      "> /tmp/ik_walk.log 2>&1 &"
    );
  }

  stopWalk();
}

void TaskMaster::onPlaying()
{
    // 장애물이 없거나 아직 멀면 그냥 직진. 멀 때 옆걸음해 버리면 다가가는 동안 판이 화면 밖으로
    // 빠져서 안 보이게 되고, 그 상태로 직진하다 부딪힘 -> 가까이 와서 (판이 잘 보일 때) 피한다
    if(!detected_ || nearest_x_m_ > avoid_dist_m_)
    {
      logDecision("STRAIGHT (far)");
      walk();
      return;
    }

    // gap_y 는 빈 틈 가운데라 딱 0 이 안 나옴 -> 이 안이면 정면으로 봄 (화면 STRAIGHT 와 같은 값)
    const float kStraightBand = 0.05;
    if(gap_found_ && -kStraightBand < std::abs(gap_y_m_) && std::abs(gap_y_m_) < kStraightBand)
    {
      logDecision("STRAIGHT");
      walk();
    }
    else if(gap_y_m_ <= -kStraightBand) //마이너스 -> 우횡진 (전부 막혀도 gap_y 쪽으로 옆걸음)
    {
      logDecision("RIGHT");
      rightwalk();
    }
    else if(kStraightBand <= gap_y_m_) // 플러스 -> 좌횡진
    {
      logDecision("LEFT");
      leftwalk();
    }
    else
    {
      logDecision("STOP");
      stopWalk();  // 전부 막혔는데 갈 쪽도 없음. 명령 안 보내면 ik_walk 가 직전 명령을 계속함
    }
}

// 보행 판단이 바뀔 때만 로그 (매 tick 찍으면 너무 많음). 명령이 자주 뒤집히는지 볼 때 씀
void TaskMaster::logDecision(const char * decision)
{
  if (last_decision_ == decision) {return;}
  last_decision_ = decision;
  RCLCPP_INFO(
    get_logger(), "walk: %s  (row0 x=%.2fm, gap_y=%+.2fm, gap_found=%d)",
    decision, nearest_x_m_, gap_y_m_, gap_found_);
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
  publishWalk(straight_cmd_);
}

void TaskMaster::leftwalk()
{
  publishWalk(left_cmd_);
}

void TaskMaster::rightwalk()
{
  publishWalk(right_cmd_);
}

void TaskMaster::publishWalk(const WalkCmd & cmd)
{
  humanoid_interfaces::msg::Master2IkMsg msg;
  msg.x_length = cmd.x;
  msg.y_length = cmd.y;
  msg.yaw = cmd.yaw;
  msg.flag = 1.0;
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