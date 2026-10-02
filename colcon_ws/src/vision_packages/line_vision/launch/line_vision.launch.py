from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = PathJoinSubstitution([FindPackageShare('line_vision'), 'config', 'line_vision.yaml'])

    # 보통 obstacle_vision 런치가 카메라를 켜므로 기본은 false. line_vision 만 따로 볼 땐 camera:=true
    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution(
            [FindPackageShare('insta360_usb_cam'), 'launch', 'usb_cam.launch.py'])),
        condition=IfCondition(LaunchConfiguration('camera')),
    )

    node = Node(
        package='line_vision',
        executable='line_vision_node',
        name='line_vision',
        output='screen',
        parameters=[config, {'image_path': LaunchConfiguration('image_path')}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('camera', default_value='false'),
        DeclareLaunchArgument('image_path', default_value='',
                              description='사진 테스트: 경로 또는 와일드카드 (따옴표로 감쌀 것)'),
        camera,
        node,
    ])
