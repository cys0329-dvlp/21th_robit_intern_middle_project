cmake_minimum_required(VERSION 3.8)
project(task_master)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(humanoid_interfaces REQUIRED)
find_package(gamecontroller REQUIRED)
# TODO(vision_interfaces): 패키지 만든 후 주석 해제
# find_package(vision_interfaces REQUIRED)

add_executable(task_master_node
  src/task_master.cpp
)
target_include_directories(task_master_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
ament_target_dependencies(task_master_node
  rclcpp
  std_msgs
  humanoid_interfaces
  gamecontroller
  # vision_interfaces  # TODO(vision_interfaces)
)

install(TARGETS task_master_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
