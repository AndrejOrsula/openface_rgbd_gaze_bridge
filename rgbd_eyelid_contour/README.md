# OpenFace RGB-D Head Pose

This package is a small extension to the [ROS 2 wrapper for OpenFace](https://github.com/AndrejOrsula/ros2_openface) that provides more robust 3D head position estimation by the use of depth stream. The orientation component of the head pose remains unchanged and is reused from OpenFace.

The `openface_rgbd_head_pose` node requires `openface_separate` to be running with *2D eye visible features* enabled. It also required a depth stream that is aligned to the color camera used in OpenFace.
