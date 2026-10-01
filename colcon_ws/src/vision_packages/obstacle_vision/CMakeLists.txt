cmake_minimum_required(VERSION 3.8)
project(obstacle_vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(vision_interfaces REQUIRED)

add_executable(obstacle_vision_node
  src/obstacle_vision_node.cpp
  src/obstacle_detector.cpp
)
target_include_directories(obstacle_vision_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
ament_target_dependencies(obstacle_vision_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  vision_interfaces
)
target_link_libraries(obstacle_vision_node ${OpenCV_LIBS})

install(TARGETS obstacle_vision_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
