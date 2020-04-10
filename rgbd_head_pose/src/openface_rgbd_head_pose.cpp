/// Extractor of eye regions based on OpenFACE features

////////////////////
/// DEPENDENCIES ///
////////////////////

// ROS 2
#include <rclcpp/rclcpp.hpp>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <cv_bridge/cv_bridge.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

// ROS 2 interfaces
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <openface_msgs/msg/pixels_stamped.hpp>

// OpenCV
#include <opencv2/highgui/highgui.hpp>

//////////////////
/// NAMESPACES ///
//////////////////

using namespace std::placeholders;

/////////////////
/// CONSTANTS ///
/////////////////

/// The name of this node
const std::string NODE_NAME = "openface_rgbd_head_pose";
/// Size of the qeueu size used by the synchronizer in its policy
const uint8_t SYNCHRONIZER_QUEUE_SIZE = 10;

/// Index of the left eye
const uint8_t EYE_LEFT = 0;
/// Index of the right eye
const uint8_t EYE_RIGHT = 1;

/// Index of the outer eye corner for the left eye
const uint8_t OUTER_CORNER_LEFT = 45;
/// Index of the outer eye corner for the right eye
const uint8_t OUTER_CORNER_RIGHT = 36;

/// Determines the neighbourhood size around eye corner that is considered to estimation of its 3D position
const uint16_t CORNER_NEIGHBOURHOOD_SIZE = 5;

/// Name of the TF frame for head
const std::string TF_FRAME_HEAD = "head";

/////////////
/// TYPES ///
/////////////

/// Policy of the synchronizer
typedef message_filters::sync_policies::ExactTime<openface_msgs::msg::PixelsStamped,
                                                  geometry_msgs::msg::PoseStamped,
                                                  sensor_msgs::msg::Image,
                                                  sensor_msgs::msg::CameraInfo>
    synchronizer_policy;

//////////////////
/// NODE CLASS ///
//////////////////

/// Class representation of this node
class OpenfaceRgbdHeadPose : public rclcpp::Node
{
public:
  /// Constructor
  OpenfaceRgbdHeadPose();

private:
  /// Subscriber to the face information obtained by OpenFace
  message_filters::Subscriber<openface_msgs::msg::PixelsStamped> sub_face_;
  /// Subscriber to the head pose obtained by OpenFace
  message_filters::Subscriber<geometry_msgs::msg::PoseStamped> sub_head_pose_;
  /// Subscriber to registered (aligned) depth frames
  image_transport::SubscriberFilter sub_depth_;
  /// Subscriber to the camera info
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_camera_info_;
  /// Synchronizer of the subscribers
  message_filters::Synchronizer<synchronizer_policy> synchronizer_;

  /// Publisher of the head pose
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_head_pose_;

  /// Broadcaster of the TF frames acquired by OpenFace
  tf2_ros::TransformBroadcaster tf2_broadcaster_;

  /// Callback called each time a message is received on all topics
  void synchronized_callback(const openface_msgs::msg::PixelsStamped::SharedPtr msg_visible_landmarks,
                             const geometry_msgs::msg::PoseStamped::SharedPtr msg_head_pose,
                             const sensor_msgs::msg::Image::SharedPtr msg_img_depth,
                             const sensor_msgs::msg::CameraInfo::SharedPtr msg_camera_info);

  /// Create a 3D cloud point at a specific pixel from depth map
  bool create_cloud_point(
      const cv::Mat_<uint16_t> &img_depth,
      cv::Point3_<double> &output,
      const std::array<double, 9> &camera_matrix,
      cv::Point_<float> &pixel,
      const double depth_scale = 0.001);
};

OpenfaceRgbdHeadPose::OpenfaceRgbdHeadPose() : Node(NODE_NAME),
                                               sub_face_(this, "openface/landmarks_visible"),
                                               sub_head_pose_(this, "openface/head_pose"),
                                               sub_depth_(this, "camera/aligned_depth_to_color/image_raw", "raw"),
                                               sub_camera_info_(this, "camera/aligned_depth_to_color/camera_info"),
                                               synchronizer_(synchronizer_policy(SYNCHRONIZER_QUEUE_SIZE), sub_face_, sub_head_pose_, sub_depth_, sub_camera_info_),
                                               tf2_broadcaster_(this)
{
  // Synchronize the subscriptions under a single callback
  synchronizer_.registerCallback(&OpenfaceRgbdHeadPose::synchronized_callback, this);

  // Create a publisher
  rclcpp::QoS qos = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
  pub_head_pose_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("openface/rgbd_head_pose", qos);

  // Declaration of the available parameters
  this->declare_parameter<bool>("broadcast_tf", true);

  RCLCPP_INFO(this->get_logger(), "Node initialised");
}

void OpenfaceRgbdHeadPose::synchronized_callback(const openface_msgs::msg::PixelsStamped::SharedPtr msg_visible_landmarks,
                                                 const geometry_msgs::msg::PoseStamped::SharedPtr msg_head_pose,
                                                 const sensor_msgs::msg::Image::SharedPtr msg_img_depth,
                                                 const sensor_msgs::msg::CameraInfo::SharedPtr msg_camera_info)
{
  RCLCPP_DEBUG(this->get_logger(), "Received face for processing");

  // Make sure the landmarks were found
  if (msg_visible_landmarks->pixels.empty())
  {
    RCLCPP_WARN(this->get_logger(), "The received face message does not contain any facial landmarks.");
    return;
  }

  // Convert color and depth images to CV
  cv_bridge::CvImagePtr img_depth;
  try
  {
    img_depth = cv_bridge::toCvCopy(msg_img_depth, sensor_msgs::image_encodings::TYPE_16UC1);
  }
  catch (const cv_bridge::Exception exception)
  {
    RCLCPP_ERROR(this->get_logger(), "Invalid frame. Exception from cv_bridge: %s", exception.what());
    return;
  }

  cv::Point3_<double> landmark_position_3d[2] = {cv::Vec<double, 3>(0.0, 0.0, 0.0), cv::Vec<double, 3>(0.0, 0.0, 0.0)};
  for (uint8_t eye = 0; eye < 2; eye++)
  {
    uint8_t landmark_index;
    if (eye == EYE_LEFT)
    {
      landmark_index = OUTER_CORNER_LEFT;
    }
    else
    {
      landmark_index = OUTER_CORNER_RIGHT;
    }

    auto landmark = msg_visible_landmarks->pixels[landmark_index];

    // Iterate over the neighbourhood and compute 3D position for each pixel
    uint8_t correct_count = 0;
    for (uint16_t r = 0; r < CORNER_NEIGHBOURHOOD_SIZE; ++r)
    {
      for (uint16_t c = 0; c < CORNER_NEIGHBOURHOOD_SIZE; ++c)
      {
        cv::Point3_<double> sample;
        cv::Point_<float> pixel = cv::Point_<double>(landmark.x - CORNER_NEIGHBOURHOOD_SIZE / 2 + c,
                                                      landmark.y - CORNER_NEIGHBOURHOOD_SIZE / 2 + r);
        if (create_cloud_point(img_depth->image, sample, msg_camera_info->k, pixel))
        {
          correct_count++;
          landmark_position_3d[eye] += sample;
        }
      }
    }
    if (correct_count > 0)
    {
      landmark_position_3d[eye] /= correct_count;
    }
    else
    {
      RCLCPP_ERROR(this->get_logger(), "Cannot obtain any depth information for the eye corner landmarks");
      return;
    }
  }

  // Find a mid-point between the eye landmarks
  cv::Point3_<double> mid_point = (landmark_position_3d[0] + landmark_position_3d[1]) / 2.0;

  // Create output msg
  geometry_msgs::msg::PoseStamped rgbd_head_pose;
  rgbd_head_pose.header = msg_head_pose->header;
  rgbd_head_pose.pose.orientation = msg_head_pose->pose.orientation;
  rgbd_head_pose.pose.position.x = mid_point.x;
  rgbd_head_pose.pose.position.y = mid_point.y;
  rgbd_head_pose.pose.position.z = mid_point.z;

  if (this->get_parameter("broadcast_tf").get_value<bool>())
  {
    geometry_msgs::msg::TransformStamped transform;
    transform.header = msg_head_pose->header;
    transform.child_frame_id = TF_FRAME_HEAD;
    transform.transform.translation.x = rgbd_head_pose.pose.position.x;
    transform.transform.translation.y = rgbd_head_pose.pose.position.y;
    transform.transform.translation.z = rgbd_head_pose.pose.position.z;
    transform.transform.rotation = rgbd_head_pose.pose.orientation;
    tf2_broadcaster_.sendTransform(transform);
  }

  pub_head_pose_->publish(rgbd_head_pose);
}

bool OpenfaceRgbdHeadPose::create_cloud_point(
    const cv::Mat_<uint16_t> &img_depth,
    cv::Point3_<double> &output,
    const std::array<double, 9> &camera_matrix,
    cv::Point_<float> &pixel,
    const double depth_scale)
{
  if (pixel.x >= img_depth.cols || pixel.y >= img_depth.rows)
  {
    RCLCPP_ERROR(this->get_logger(), "create_cloud_point() - Pixel out of bounds");
    return false;
  }

  // Get subpixel value of the depth at floating point pixel coordinate
  cv::Mat subpixel_patch;
  cv::remap(img_depth, subpixel_patch, cv::Mat(1, 1, CV_32FC2, &pixel), cv::noArray(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
  uint16_t z = subpixel_patch.at<uint16_t>(0, 0);

  if (z == 0)
  {
    return false;
  }
  else
  {
    output.z = z * depth_scale;
    output.x = output.z * ((pixel.x - camera_matrix[2]) / camera_matrix[0]);
    output.y = output.z * ((pixel.y - camera_matrix[5]) / camera_matrix[4]);
    return true;
  }
}

////////////
/// MAIN ///
////////////

/// Main function that initiates an object of `OpenfaceRgbdHeadPose` class as the core of this node.
int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OpenfaceRgbdHeadPose>());
  rclcpp::shutdown();
  return EXIT_SUCCESS;
}
