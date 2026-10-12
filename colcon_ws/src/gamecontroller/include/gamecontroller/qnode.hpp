/**
 * @file /include/gamecontroller/qnode.hpp
 *
 * @brief Communications central!
 *
 * @date February 2011
 **/
/*****************************************************************************
** Ifdefs
*****************************************************************************/

#ifndef gamecontroller_QNODE_HPP_
#define gamecontroller_QNODE_HPP_

/*****************************************************************************
** Includes
*****************************************************************************/
#ifndef Q_MOC_RUN
#include <rclcpp/rclcpp.hpp>
#endif
#include <QThread>
#include <QString>
#include <string>

#include "std_msgs/msg/string.hpp"

#include "gamecontroller/msg/robocupcontroller.hpp"
#include "gamecontroller/msg/gamecontroldata.hpp"
#include "gamecontroller/msg/gamecontrolreturndata.hpp"

/*****************************************************************************
** Class
*****************************************************************************/


class QNode : public QThread
{
  Q_OBJECT
public:
  QNode(int argc, char** argv);
  ~QNode();

  void run();

  gamecontroller::msg::Gamecontroldata gameControlData;
  gamecontroller::msg::Gamecontrolreturndata gameControlReturnData;

  rclcpp::Publisher<gamecontroller::msg::Gamecontroldata>::SharedPtr gamecontrollerPub;
  // task_master 가 보행 판단이 바뀔 때마다 보내는 로그
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr walkDecisionSub;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr lineStatusSub;  // 0.1초마다 오는 상태 줄

  // config.yaml 파라미터
  std::string pubTopic;
  std::string walkDecisionTopic;
  std::string lineStatusTopic;
  int teamRobit;
  int teamRobitRed;
  int teamRobitBlue;
  int dataPort;
  int returnPort;

private:
  std::shared_ptr<rclcpp::Node> node;

Q_SIGNALS:
  void rosShutDown();
  // ROS 스레드에서 받은 걸 UI 스레드로 넘김 (여기서 ui 를 직접 건드리면 안 됨)
  void walkDecision(QString text);
  void lineStatus(QString text);
};

#endif /* gamecontroller_QNODE_HPP_ */
