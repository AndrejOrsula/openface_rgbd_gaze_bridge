"""Launch OpenFace with eye region extraction"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir


def generate_launch_description():

    config_openface_separate = LaunchConfiguration('config_openface_separate', default=os.path.join(get_package_share_directory(
        'eye_region'), 'config', 'openface_separate.yaml'))

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_openface_separate',
            default_value=config_openface_separate,
            description='Path to config for openface'),

        Node(
            package='openface',
            node_executable='openface_separate',
            node_name='openface_separate',
            node_namespace='',
            output='screen',
            parameters=[config_openface_separate],
            remappings=[('camera/image_raw', 'camera/color/image_raw'),
                        ('camera/camera_info', 'camera/color/camera_info'),
                        ('openface/landmarks_visible', 'openface/landmarks_visible')],
        ),

        Node(
            package='openface_eye_region',
            node_executable='openface_eye_region',
            node_name='openface_eye_region',
            node_namespace='',
            output='screen',
            remappings=[('openface/landmarks_visible', 'openface/landmarks_visible'),
                        ('openface/eye_regions', 'openface/eye_regions')],
        ),
    ])
