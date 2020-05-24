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

// ROS 2 interfaces
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <openface_msgs/msg/pixels_stamped.hpp>
#include <eyelid_contour_msgs/msg/eyelid_contours_stamped.hpp>

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
const std::string NODE_NAME = "openface_rgbd_eyelid_contour";
/// Size of the qeueu size used by the synchronizer in its policy
const uint8_t SYNCHRONIZER_QUEUE_SIZE = 10;

/// Index of the left eye
const uint8_t EYE_LEFT = 0;
/// Index of the right eye
const uint8_t EYE_RIGHT = 1;

/// Number of eye landmarks per each eye
const uint8_t LANDMARKS_COUNT = 12;
/// Index of the start for the left eye landmarks
const uint8_t LANDMARKS_OFFSET_LEFT = 36;
/// Index of the start for the right eye landmarks
const uint8_t LANDMARKS_OFFSET_RIGHT = 8;

/// Flag that determines whether to skip inner corners of the eyes, which can cause distortion the spherical eyeball nature due to caruncle
const bool SKIP_INNER_CORNERS = false;
/// Indecis of the inner eye corners that are currently ignored. Set to 0s to enable outputting of these landmarks.
const uint8_t LANDMARKS_INNER_CORNERS[] = {36, 14};

/// Determines the neighbourhood size around eye corner that is considered to estimation of its 3D position
const uint16_t CORNER_NEIGHBOURHOOD_SIZE = 3;

/////////////
/// TYPES ///
/////////////

/// Policy of the synchronizer
typedef message_filters::sync_policies::ExactTime<openface_msgs::msg::PixelsStamped,
                                                  sensor_msgs::msg::Image,
                                                  sensor_msgs::msg::CameraInfo>
    synchronizer_policy;

//////////////////
/// NODE CLASS ///
//////////////////

/// Class representation of this node
class OpenfaceRgbdEyelidContour : public rclcpp::Node
{
public:
  /// Constructor
  OpenfaceRgbdEyelidContour();

private:
  /// Subscriber to the face information obtained by OpenFace
  message_filters::Subscriber<openface_msgs::msg::PixelsStamped> sub_face_;
  /// Subscriber to registered (aligned) depth frames
  image_transport::SubscriberFilter sub_depth_;
  /// Subscriber to the camera info
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> sub_camera_info_;
  /// Synchronizer of the subscribers
  message_filters::Synchronizer<synchronizer_policy> synchronizer_;

  /// Publisher of the head pose
  rclcpp::Publisher<eyelid_contour_msgs::msg::EyelidContoursStamped>::SharedPtr pub_eyelid_contours_;

  /// Callback called each time a message is received on all topics
  void synchronized_callback(const openface_msgs::msg::PixelsStamped::SharedPtr msg_visible_eye_landmarks,
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

OpenfaceRgbdEyelidContour::OpenfaceRgbdEyelidContour() : Node(NODE_NAME),
                                                         sub_face_(this, "openface/eye_landmarks_visible"),
                                                         sub_depth_(this, "camera/aligned_depth_to_color/image_raw", "raw"),
                                                         sub_camera_info_(this, "camera/aligned_depth_to_color/camera_info"),
                                                         synchronizer_(synchronizer_policy(SYNCHRONIZER_QUEUE_SIZE), sub_face_, sub_depth_, sub_camera_info_)
{
  // Synchronize the subscriptions under a single callback
  synchronizer_.registerCallback(&OpenfaceRgbdEyelidContour::synchronized_callback, this);

  // Create a publisher
  rclcpp::QoS qos = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
  pub_eyelid_contours_ = this->create_publisher<eyelid_contour_msgs::msg::EyelidContoursStamped>("openface/eyelid_contours", qos);

  RCLCPP_INFO(this->get_logger(), "Node initialised");
}

void OpenfaceRgbdEyelidContour::synchronized_callback(const openface_msgs::msg::PixelsStamped::SharedPtr msg_visible_eye_landmarks,
                                                      const sensor_msgs::msg::Image::SharedPtr msg_img_depth,
                                                      const sensor_msgs::msg::CameraInfo::SharedPtr msg_camera_info)
{
  RCLCPP_DEBUG(this->get_logger(), "Received face for processing");

  // Make sure the landmarks were found
  if (msg_visible_eye_landmarks->pixels.empty())
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

  // Create output msg
  eyelid_contour_msgs::msg::EyelidContoursStamped eyelid_contours;
  eyelid_contours.header = msg_visible_eye_landmarks->header;

  // Fill the message with 3D position of eylid landmarks for both eyes
  for (uint8_t eye = 0; eye < 2; eye++)
  {
    uint8_t landmarks_offset;
    if (eye == EYE_LEFT)
    {
      landmarks_offset = LANDMARKS_OFFSET_LEFT;
    }
    else
    {
      landmarks_offset = LANDMARKS_OFFSET_RIGHT;
    }

    for (uint8_t landmark = landmarks_offset; landmark < landmarks_offset + LANDMARKS_COUNT; landmark++)
    {
      cv::Point3_<double> contour_point;

      // Skip inner corners, if enabled
      if (SKIP_INNER_CORNERS && landmark == LANDMARKS_INNER_CORNERS[eye])
      {
        continue;
      }

      // Iterate over the neighbourhood and compute 3D position for each pixel
      uint8_t correct_count = 0;
      for (uint16_t r = 0; r < CORNER_NEIGHBOURHOOD_SIZE; ++r)
      {
        for (uint16_t c = 0; c < CORNER_NEIGHBOURHOOD_SIZE; ++c)
        {
          cv::Point3_<double> sample;
          cv::Point_<float> pixel = cv::Point_<double>(msg_visible_eye_landmarks->pixels[landmark].x - CORNER_NEIGHBOURHOOD_SIZE / 2 + c,
                                                       msg_visible_eye_landmarks->pixels[landmark].y - CORNER_NEIGHBOURHOOD_SIZE / 2 + r);
          if (create_cloud_point(img_depth->image, sample, msg_camera_info->k, pixel))
          {
            correct_count++;
            contour_point += sample;
          }
        }
      }
      if (correct_count > 0)
      {
        contour_point /= correct_count;

        geometry_msgs::msg::Point contour_point_msg;
        contour_point_msg.x = contour_point.x;
        contour_point_msg.y = contour_point.y;
        contour_point_msg.z = contour_point.z;
        eyelid_contours.eyes.eyelids[eye].contour.push_back(contour_point_msg);
      }
    }
  }

  pub_eyelid_contours_->publish(eyelid_contours);
}

bool OpenfaceRgbdEyelidContour::create_cloud_point(
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

/// Main function that initiates an object of `OpenfaceRgbdEyelidContour` class as the core of this node.
int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OpenfaceRgbdEyelidContour>());
  rclcpp::shutdown();
  return EXIT_SUCCESS;
}
