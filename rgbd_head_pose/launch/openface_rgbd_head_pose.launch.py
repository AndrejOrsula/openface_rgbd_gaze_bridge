"""Launch OpenFace with RGB-D head pose"""

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
        'openface_rgbd_head_pose'), 'config', 'openface_separate.yaml'))
    config_openface_rgbd_head_pose = LaunchConfiguration('config_openface_rgbd_head_pose', default=os.path.join(get_package_share_directory(
        'openface_rgbd_head_pose'), 'config', 'openface_rgbd_head_pose.yaml'))

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_openface_separate',
            default_value=config_openface_separate,
            description='Path to config for openface'),
        DeclareLaunchArgument(
            'config_openface_rgbd_head_pose',
            default_value=config_openface_rgbd_head_pose,
            description='Path to config for RGB-D head pose extractor'),

        Node(
            package='openface',
            node_executable='openface_separate',
            node_name='openface_separate',
            node_namespace='',
            output='screen',
            parameters=[config_openface_separate],
            remappings=[('camera/image_raw', 'camera/color/image_raw'),
                        ('camera/camera_info', 'camera/color/camera_info'),
                        ('openface/landmarks_visible',
                         'openface/landmarks_visible'),
                        ('openface/head_pose', 'openface/head_pose')],
        ),

        Node(
            package='openface_rgbd_head_pose',
            node_executable='openface_rgbd_head_pose',
            node_name='openface_rgbd_head_pose',
            node_namespace='',
            output='screen',
            parameters=[config_openface_rgbd_head_pose],
            remappings=[('openface/landmarks_visible', 'openface/landmarks_visible'),
                        ('openface/head_pose', 'openface/head_pose'),
                        ('camera/aligned_depth_to_color/image_raw',
                         'camera/aligned_depth_to_color/image_raw'),
                        ('camera/aligned_depth_to_color/camera_info',
                         'camera/aligned_depth_to_color/camera_info'),
                        ('openface/rgbd_head_pose', 'openface/rgbd_head_pose')],
        ),
    ])
