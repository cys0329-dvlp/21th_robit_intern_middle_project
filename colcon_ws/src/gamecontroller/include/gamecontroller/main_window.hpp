/**
 * @file /include/gamecontroller/main_window.hpp
 *
 * @brief Qt based gui for %(package)s.
 *
 * @date August 2026
 **/

#ifndef gamecontroller_MAIN_WINDOW_H
#define gamecontroller_MAIN_WINDOW_H

/*****************************************************************************
** Includes
*****************************************************************************/

#include <QMainWindow>
#include "QIcon"
#include "qnode.hpp"
#include "ui_mainwindow.h"

#include "RoboCupGameControlData.h"

#include <QtNetwork>
#include <QUdpSocket>
#include <QHostAddress>
#include <QTimer>
#include <QTime>
#include <QStringList>
#include <QProcess>
#include <QMap>

#include <iostream>

#include "QKeyEvent"
#include "technical_dialog.hpp"

// #define TEAM_ROBIT 4 //    TEAM-ROBIT_NUMBER: 4 CONFIG.YAML
// #define TEAM_ROBIT_RED 22
// #define TEAM_ROBIT_BLUE 23

#define LEFT 0
#define RIGHT 1

#define POSITION 0
#define POSITION_GK 1
#define POSITION_FW 2
#define POSITION_TECHNICAL 3

#define STATE_TECHNICAL 0
#define STATE_TCCL_RK 1
#define STATE_TCCL_RH 2
#define STATE_TCDK 3
#define STATE_TCHK 4
#define STATE_TCPK 5
#define STATE_TCOB 5

using namespace std;
/*****************************************************************************
** Interface [MainWindow]
*****************************************************************************/
/**
 * @brief Qt central, all operations relating to the view part here.
 */
class MainWindow : public QMainWindow
{
  Q_OBJECT

public:
  QTimer *mTimer;
  MainWindow(int argc, char** argv, QWidget *parent = nullptr);
  ~MainWindow();
  QNode *qnode;

  bool initAddrAndPort();
  void initSocket();
  void closeSocket();

  // kjh edit
  void uiUpdate();
  void redrawText();
  void ChangeTechnicalMode();

  QList<QHostAddress> ipAddressesList;

private:
  Ui::RobocupController *ui;
  void closeEvent(QCloseEvent *event);

  QTimer *g_timer;
  QTimer *pub_timer;
  
  QHostAddress senderAddress;
  quint16 senderPort;

  QUdpSocket *m_pReadSocket = NULL;
  QUdpSocket *m_pSendSocket = NULL;

  QHostAddress m_qSrcAddress;
  QHostAddress m_qDestAddress;

  quint16 m_iPort;
  QString m_qstrIp;

  bool m_bIsServerOpen = false;

  int position = 0;
  int playerNum = 0;
  int initSide = 0;
  int mySide = 0;
  int myTeam = 4;

  // textEdit 아래에 붙이는 task_master 보행 판단 로그 (최근 WALK_LOG_MAX 줄)
  static const int WALK_LOG_MAX = 8;
  QStringList walkLog;
  QString lineStatusText;    // 0.1초마다 덮어쓰는 상태 줄 (로그처럼 쌓지 않음)
  bool hasGameData = false;  // GC 패킷을 한 번이라도 받았는지 (안 받았으면 robocupData 가 비어 있음)

  // 버튼으로 켜는 ros2 launch 프로세스 (key -> QProcess)
  QMap<QString, QProcess*> launchProcs;
  void toggleLaunch(const QString& key, const QStringList& args, bool on);

private Q_SLOTS:
  void on_comboBox_position_currentIndexChanged(int index);
  void on_Server_Open_clicked();
  void GameController_callback();

  void readData();
  void Pub_msg();
  void onWalkDecision(QString text);
  void onLineStatus(QString text);
  void on_comboBox_state_currentIndexChanged(int index);

  void on_btn_line_vision_toggled(bool on);
  void on_btn_obstacle_vision_toggled(bool on);
  void on_btn_imu_toggled(bool on);
  void on_btn_task_master_toggled(bool on);

private:
  struct RoboCupGameControlData robocupData;
  struct RoboCupGameControlReturnData robocupreturnData;
  struct RobotInfo robotInfo;
  struct TeamInfo teamInfo;
};

#endif // gamecontroller_MAIN_WINDOW_H
