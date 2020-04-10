/// Extractor of eye regions based on OpenFACE features

////////////////////
/// DEPENDENCIES ///
////////////////////

// ROS 2
#include <rclcpp/rclcpp.hpp>

// ROS 2 interfaces
#include <openface_msgs/msg/pixels_stamped.hpp>
#include <eye_region_msgs/msg/eye_regions_of_interest_stamped.hpp>

// C++
#include <math.h>

//////////////////
/// NAMESPACES ///
//////////////////

using namespace std::placeholders;

/////////////////
/// CONSTANTS ///
/////////////////

/// The name of this node
const std::string NODE_NAME = "openface_eye_region";

/// Index of the left eye
const uint8_t EYE_LEFT = 0;
/// Index of the right eye
const uint8_t EYE_RIGHT = 1;

/// Number of eye landmarks per each eye
const uint8_t LANDMARKS_COUNT = 6;
/// Index of the start for the left eye landmarks
const uint8_t LANDMARKS_OFFSET_LEFT = 42;
/// Index of the start for the right eye landmarks
const uint8_t LANDMARKS_OFFSET_RIGHT = 36;

//////////////////
/// NODE CLASS ///
//////////////////

/// Class representation of this node
class OpenfaceEyeRegion : public rclcpp::Node
{
public:
  /// Constructor
  OpenfaceEyeRegion();

private:
  /// Subscriber to the face information obtained by OpenFace
  rclcpp::Subscription<openface_msgs::msg::PixelsStamped>::SharedPtr sub_face_;
  /// Publisher of eye ROIs
  rclcpp::Publisher<eye_region_msgs::msg::EyeRegionsOfInterestStamped>::SharedPtr pub_eyes_;

  /// Callback called each time openface finishes analysis of a face and publishes visible landmarks
  void face_callback(const openface_msgs::msg::PixelsStamped::SharedPtr visible_landmarks);
};

OpenfaceEyeRegion::OpenfaceEyeRegion() : Node(NODE_NAME)
{
  // Create a subscriber to face from Openface
  rclcpp::QoS qos_face = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
  sub_face_ = this->create_subscription<openface_msgs::msg::PixelsStamped>("openface/landmarks_visible", qos_face, std::bind(&OpenfaceEyeRegion::face_callback, this, _1));

  // Create a publisher
  rclcpp::QoS qos_eyes = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
  pub_eyes_ = this->create_publisher<eye_region_msgs::msg::EyeRegionsOfInterestStamped>("openface/eye_regions", qos_eyes);

  RCLCPP_INFO(this->get_logger(), "Node initialised");
}

void OpenfaceEyeRegion::face_callback(const openface_msgs::msg::PixelsStamped::SharedPtr visible_landmarks)
{
  RCLCPP_DEBUG(this->get_logger(), "Received face for processing");

  // Make sure the landmarks were found
  if (visible_landmarks->pixels.empty())
  {
    RCLCPP_WARN(this->get_logger(), "The received face message does not contain any facial landmarks.");
    return;
  }

  // Find eye ROIs
  float top_left_corners[2][2] = {{std::numeric_limits<float>::max(), std::numeric_limits<float>::max()},
                                  {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()}};
  float bottom_right_corners[2][2] = {{0, 0},
                                      {0, 0}};
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
      // Horizontal coordinates
      if (visible_landmarks->pixels[landmark].x < top_left_corners[0][eye])
      {
        top_left_corners[0][eye] = visible_landmarks->pixels[landmark].x;
      }
      else if (visible_landmarks->pixels[landmark].x > bottom_right_corners[0][eye])
      {
        bottom_right_corners[0][eye] = visible_landmarks->pixels[landmark].x;
      }

      // Vertical coordinates
      if (visible_landmarks->pixels[landmark].y < top_left_corners[1][eye])
      {
        top_left_corners[1][eye] = visible_landmarks->pixels[landmark].y;
      }
      else if (visible_landmarks->pixels[landmark].y > bottom_right_corners[1][eye])
      {
        bottom_right_corners[1][eye] = visible_landmarks->pixels[landmark].y;
      }
    }
  }

  // Create a message containing the eye ROIs, with the same header as the received message
  eye_region_msgs::msg::EyeRegionsOfInterestStamped eyes;
  eyes.header = visible_landmarks->header;

  for (uint8_t eye = 0; eye < 2; eye++)
  {
    eyes.eyes.rois[eye].x_offset = std::floor(top_left_corners[0][eye]);
    eyes.eyes.rois[eye].y_offset = std::floor(top_left_corners[1][eye]);
    eyes.eyes.rois[eye].width = std::ceil(bottom_right_corners[0][eye]) - eyes.eyes.rois[eye].x_offset;
    eyes.eyes.rois[eye].height = std::ceil(bottom_right_corners[1][eye]) - eyes.eyes.rois[eye].y_offset;
  }

  pub_eyes_->publish(eyes);
}

////////////
/// MAIN ///
////////////

/// Main function that initiates an object of `OpenfaceEyeRegion` class as the core of this node.
int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OpenfaceEyeRegion>());
  rclcpp::shutdown();
  return EXIT_SUCCESS;
}
