from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution(
        [FindPackageShare('obstacle_vision'), 'config', 'obstacle_vision.yaml'])

    # camera:=false 로 끄면 카메라 노드 없이 (사진 테스트 등) 실행
    # FindPackageShare 는 실제로 include 할 때만 찾으므로, 꺼두면 insta360_usb_cam 이 없어도 된다
    use_camera = LaunchConfiguration('camera')

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('insta360_usb_cam'), 'launch', 'usb_cam.launch.py'])),
        condition=IfCondition(use_camera),
    )

    vision = Node(
        package='obstacle_vision',
        executable='obstacle_vision_node',
        name='obstacle_vision',
        output='screen',
        parameters=[config],
    )

    return LaunchDescription([
        DeclareLaunchArgument('camera', default_value='true'),
        camera,
        vision,
    ])
