#include <ros/ros.h>
#include <ros/package.h>
#include <stdio.h>
#include <unistd.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <memory>
#include <tuple>
#include <set>
#include <sstream>
#include <thread>
#include <atomic>
#include <mutex>
#include <future>
#include <exception>

#include <tbb/tbb.h>

#include <opencv2/videoio/videoio.hpp>
#include <opencv2/videoio/videoio_c.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "openface2_ros_msgs/ActionUnit.h"
#include "openface2_ros_msgs/Face.h"
#include "openface2_ros_msgs/Faces.h"

#include <sensor_msgs/Image.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"
#include "Visualizer.h"
#include "VisualizationUtils.h"

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>

using namespace std;
using namespace ros;
using namespace cv;

namespace
{
  static geometry_msgs::Quaternion toQuaternion(double pitch, double roll, double yaw)
  {
    double t0 = std::cos(yaw * 0.5f);
    double t1 = std::sin(yaw * 0.5f);
    double t2 = std::cos(roll * 0.5f);
    double t3 = std::sin(roll * 0.5f);
    double t4 = std::cos(pitch * 0.5f);
    double t5 = std::sin(pitch * 0.5f);

    geometry_msgs::Quaternion q;
    q.w = t0 * t2 * t4 + t1 * t3 * t5;
    q.x = t0 * t3 * t4 - t1 * t2 * t5;
    q.y = t0 * t2 * t5 + t1 * t3 * t4;
    q.z = t1 * t2 * t4 - t0 * t3 * t5;
    return q;
  }

  static geometry_msgs::Quaternion operator *(const geometry_msgs::Quaternion &a, const geometry_msgs::Quaternion &b)
  {
    geometry_msgs::Quaternion q;
    
    q.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;  // 1
    q.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;  // i
    q.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;  // j
    q.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;  // k
    return q;
  }

 void NonOverlapingDetections(const vector<LandmarkDetector::CLNF>& clnf_models, vector<cv::Rect_<float> >& face_detections)
 {
    // Go over the model and eliminate detections that are not informative (there already is a tracker there)
    for (size_t model = 0; model < clnf_models.size(); ++model)
    {

      // See if the detections intersect
      cv::Rect_<float> model_rect = clnf_models[model].GetBoundingBox();

      for (int detection = face_detections.size() - 1; detection >= 0; --detection)
      {
        double intersection_area = (model_rect & face_detections[detection]).area();
        double union_area = model_rect.area() + face_detections[detection].area() - 2 * intersection_area;

        // If the model is already tracking what we're detecting ignore the detection, this is determined by amount of overlap
        if (intersection_area / union_area > 0.5)
        {
          face_detections.erase(face_detections.begin() + detection);
        }
      }
    }
  }
}


namespace openface2_ros
{
  class OpenFace2Ros
  {
  public:
    OpenFace2Ros(NodeHandle &nh)
      : nh_(nh)
      , it_(nh_)
      , visualizer(true, false, false, true)
    {
      NodeHandle pnh("~");

      if(!pnh.getParam("image_topic", image_topic_)) pnh.param<string>("image_topic", image_topic_, "/realsense_face/color/image_raw");
      
      const auto base_path = package::getPath("openface2_ros");

      //pnh.param<bool>("publish_viz", publish_viz_, false);

      //if(!pnh.getParam("max_faces", max_faces_)) pnh.param<int>("max_faces", max_faces_, 4);
      //if(max_faces_ <= 0) throw invalid_argument("~max_faces must be > 0");
      max_faces_ = 1;

      //float rate = 0;
      //if(!pnh.getParam("rate", rate)) pnh.param<float>("rate", rate, 4.0);
      //if(rate <= 0) throw invalid_argument("~rate must be > 0");
      //rate_ = round(30/rate);

      camera_sub_ = it_.subscribeCamera(image_topic_, 1, &OpenFace2Ros::process_incoming_, this);
      faces_pub_ = nh_.advertise<openface2_ros_msgs::Faces>("openface2/faces", 10);
      viz_pub_ = it_.advertise("openface2/image", 1);
      init_openface_();
    }
    
    ~OpenFace2Ros()
    {
    }
    
  private:
    void init_openface_()
    {
      	vector<string> arguments(1,"");
      	LandmarkDetector::FaceModelParameters det_params(arguments);
      	// This is so that the model would not try re-initialising itself
      	//det_params.reinit_video_every = -1;

      	det_params.curr_face_detector = LandmarkDetector::FaceModelParameters::MTCNN_DETECTOR;

      	det_parameters.push_back(det_params);

      	LandmarkDetector::CLNF face_model(det_parameters[0].model_location);

      	if (!face_model.loaded_successfully)
      	{
        	cout << "ERROR: Could not load the landmark detector" << endl;
      	}

      	// Loading the face detectors
      	face_model.face_detector_HAAR.load(det_parameters[0].haar_face_detector_location);
      	face_model.haar_face_detector_location = det_parameters[0].haar_face_detector_location;
      	face_model.face_detector_MTCNN.Read(det_parameters[0].mtcnn_face_detector_location);
      	face_model.mtcnn_face_detector_location = det_parameters[0].mtcnn_face_detector_location;

      	// If can't find MTCNN face detector, default to HOG one
      	if (det_parameters[0].curr_face_detector == LandmarkDetector::FaceModelParameters::MTCNN_DETECTOR && face_model.face_detector_MTCNN.empty())
     	{
        	cout << "INFO: defaulting to HOG-SVM face detector" << endl;
        	det_parameters[0].curr_face_detector = LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR;
      	}

      	face_models.reserve(max_faces_);

      	face_models.push_back(face_model);
      	active_models.push_back(false);

      	for (int i = 1; i < max_faces_; ++i)
      	{
        	face_models.push_back(face_model);
        	active_models.push_back(false);
        	det_parameters.push_back(det_params);
      	}

      	if (!face_model.eye_model)
      	{
        	cout << "WARNING: no eye model found" << endl;
      	}

        fps_tracker.AddFrame();

        ROS_INFO("OpenFace initialized!");
    }

    void process_incoming_(const sensor_msgs::ImageConstPtr &img, const sensor_msgs::CameraInfoConstPtr &cam)
    {
        
        if(viz_pub_.getNumSubscribers() > 0 ) {
            publish_viz_ = true;
        } else {
            publish_viz_ = false;
        }


        cv_bridge::CvImagePtr cv_ptr_rgb;
        cv_bridge::CvImagePtr cv_ptr_mono;
        try
        {
        	cv_ptr_rgb = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
        	cv_ptr_mono = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
      	}
      	catch(const cv_bridge::Exception &e)
      	{
        	ROS_ERROR("cv_bridge exception: %s", e.what());
        	return;
      	}

        double fx = cam->K[0];
        double fy = cam->K[4];
        double cx = cam->K[2];
        double cy = cam->K[5];


        if(fx == 0 || fy == 0)
        {
        	fx = 500.0 * cv_ptr_rgb->image.cols / 640.0;
        	fy = 500.0 * cv_ptr_rgb->image.rows / 480.0;
        	fx = (fx + fy) / 2.0;
        	fy = fx;
        }

        if(cx == 0) cx = cv_ptr_rgb->image.cols / 2.0;
        if(cy == 0) cy = cv_ptr_rgb->image.rows / 2.0;
        int model = 0;

	bool detection_success = LandmarkDetector::DetectLandmarksInVideo(cv_ptr_rgb->image, face_models[model], det_parameters[model], cv_ptr_mono->image);


        // Keeping track of FPS
  	    fps_tracker.AddFrame();

        decltype(cv_ptr_rgb->image) viz_img = cv_ptr_rgb->image.clone();
        if(publish_viz_) visualizer.SetImage(viz_img, fx, fy, cx, cy);

        openface2_ros_msgs::Faces faces;
        faces.header.frame_id = img->header.frame_id;
        faces.header.stamp = Time::now();
        if(detection_success) {
 
				// Estimate head pose and eye gaze
	            openface2_ros_msgs::Face face;

          	    // Estimate head pose and eye gaze				
			    cv::Vec6d head_pose = LandmarkDetector::GetPose(face_models[model], fx, fy, cx, cy);
	            face.head_pose.position.x = head_pose[0] / 1000.0;
	            face.head_pose.position.y = head_pose[1] / 1000.0;
	            face.head_pose.position.z = head_pose[2] / 1000.0;
	          
	            const auto head_orientation = toQuaternion(head_pose[4], -head_pose[3], -head_pose[5]);
	            face.head_pose.orientation = toQuaternion(M_PI,  0,  0);//toQuaternion(M_PI / 2, 0, M_PI / 2);// toQuaternion(0, 0, 0);
	            face.head_pose.orientation = face.head_pose.orientation * head_orientation;

	            // tf
	            geometry_msgs::TransformStamped transform;
	            transform.header = faces.header;
	            stringstream out;
	            out << "head" << model;
	            transform.child_frame_id = out.str();
	            transform.transform.translation.x = face.head_pose.position.x;
	            transform.transform.translation.y = face.head_pose.position.y;
	            transform.transform.translation.z = face.head_pose.position.z;
	            transform.transform.rotation = face.head_pose.orientation;
	            tf_br_.sendTransform(transform);
          
          	    const std::vector<cv::Point3f> eye_landmarks3d = LandmarkDetector::Calculate3DEyeLandmarks(face_models[model], fx, fy, cx, cy);
			    cv::Point3f gaze_direction0(0, 0, 0); cv::Point3f gaze_direction1(0, 0, 0); cv::Vec2d gaze_angle(0, 0);


          		if(publish_viz_)
          		{
            		visualizer.SetObservationLandmarks(face_models[model].detected_landmarks, face_models[model].detection_certainty);
            		visualizer.SetObservationPose(LandmarkDetector::GetPose(face_models[model], fx, fy, cx, cy), face_models[model].detection_certainty);
          		}
          		//ROS_INFO("models %lu active", model);
          		faces.faces.push_back(face);
        	
        
  		//ROS_INFO("faces size %d", faces.face.size());
  		faces.count = (int)faces.faces.size();

		// we only publish faces if we have any. this way a simple rostopic hz reveals whether faces are found
  		if (faces.count > 0) faces_pub_.publish(faces);
        }
      	if(publish_viz_)
      	{
        	auto viz_msg = cv_bridge::CvImage(img->header, "bgr8", visualizer.GetVisImage()).toImageMsg();
        	viz_pub_.publish(viz_msg);
        }
    }
    
    tf2_ros::TransformBroadcaster tf_br_;
    
    // The modules that are being used for tracking
	vector<LandmarkDetector::CLNF> face_models;
	vector<bool> active_models;
    vector<LandmarkDetector::FaceModelParameters> det_parameters;

    Utilities::Visualizer visualizer;
    Utilities::FpsTracker fps_tracker;

    string image_topic_;
    int max_faces_;
    unsigned rate_;

    bool published_markers;
    NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::CameraSubscriber camera_sub_;
    Publisher faces_pub_;

    bool publish_viz_;
    image_transport::Publisher viz_pub_;
  };
}

int main(int argc, char *argv[])
{
  init(argc, argv, "openface2_ros");
  
  using namespace openface2_ros;

  NodeHandle nh;

  try
  {
    OpenFace2Ros openface_(nh);
    MultiThreadedSpinner().spin();
  }
  catch(const exception &e)
  {
    ROS_FATAL("%s", e.what());
    return EXIT_FAILURE;
  }
  
  return EXIT_SUCCESS; 
}
