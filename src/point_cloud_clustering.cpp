#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/conversions.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/TransformStamped.h>
#include <rosbag/bag.h>
#include <random>

typedef pcl::PointXYZRGB PointT;

class PointCloudClustering
{
public:
  PointCloudClustering()
    : tf_listener_(tf_buffer_)
  {
    // Parameters for the ground plane limits
    nh_.param("ground_min_z", ground_min_z_, -2.0);
    nh_.param("ground_max_z", ground_max_z_, 3.0);

    sub_ = nh_.subscribe("/me5413/lidar_top", 1, &PointCloudClustering::pointCloudCallback, this);
    pub_ = nh_.advertise<sensor_msgs::PointCloud2>("clustered_cloud", 1);
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("cluster_markers", 1);
    debug_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("debug_cloud", 1);
    // bag_.open("clustered_output.bag", rosbag::bagmode::Write);
  }

  ~PointCloudClustering()
  {
    // bag_.close();
  }

  void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    // Apply pass-through filter to limit the height of points considered for ground removal
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(ground_min_z_, ground_max_z_);
    pass.filter(*filtered_cloud);

    // Debug: Publish and output size of filtered cloud
    publishDebugCloud(filtered_cloud, "filtered_cloud");
    ROS_INFO("Filtered cloud size: %lu", filtered_cloud->points.size());

    // Segment the ground plane using RANSAC
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr ground_inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(10000);
    seg.setDistanceThreshold(0.1);
    seg.setInputCloud(filtered_cloud);
    seg.segment(*ground_inliers, *coefficients);

    if (ground_inliers->indices.empty())
    {
      ROS_WARN("Could not estimate a planar model for the given dataset.");
      return;
    }

    // Debug: Output number of inliers
    ROS_INFO("Number of ground inliers: %lu", ground_inliers->indices.size());

    // Extract the ground plane from the original cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(filtered_cloud);
    extract.setIndices(ground_inliers);
    extract.setNegative(true);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground(new pcl::PointCloud<pcl::PointXYZ>());
    extract.filter(*cloud_no_ground);

    // Debug: Publish and output size of cloud after removing ground
    publishDebugCloud(cloud_no_ground, "cloud_no_ground");
    ROS_INFO("Cloud size after ground removal: %lu", cloud_no_ground->points.size());

    // Apply a StatisticalOutlierRemoval filter to remove outliers
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground_filtered(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud_no_ground);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*cloud_no_ground_filtered);

    // Debug: Publish and output size of cloud after outlier removal
    publishDebugCloud(cloud_no_ground_filtered, "cloud_no_ground_filtered");
    ROS_INFO("Cloud size after outlier removal: %lu", cloud_no_ground_filtered->points.size());

    // Perform Euclidean clustering on the remaining points
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(cloud_no_ground_filtered);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.3);
    ec.setMinClusterSize(1000);
    ec.setMaxClusterSize(10000);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud_no_ground_filtered);
    ec.extract(cluster_indices);

    pcl::PointCloud<PointT>::Ptr clustered_cloud(new pcl::PointCloud<PointT>());
    clustered_cloud->header = cloud->header;

    visualization_msgs::MarkerArray marker_array;
    int cluster_id = 0;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (const auto& indices : cluster_indices)
    {
      pcl::PointCloud<PointT>::Ptr cluster(new pcl::PointCloud<PointT>());
      float r = dis(gen) * 255;
      float g = dis(gen) * 255;
      float b = dis(gen) * 255;

      for (const auto& index : indices.indices)
      {
        pcl::PointXYZ point = cloud_no_ground_filtered->points[index];
        PointT colored_point;
        colored_point.x = point.x;
        colored_point.y = point.y;
        colored_point.z = point.z;
        colored_point.r = r;
        colored_point.g = g;
        colored_point.b = b;
        cluster->points.push_back(colored_point);
      }

      *clustered_cloud += *cluster;

      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*cluster, centroid);

      Eigen::Vector4f min_point, max_point;
      pcl::getMinMax3D(*cluster, min_point, max_point);

      // Create a marker for the cluster
      visualization_msgs::Marker marker;
      marker.header.frame_id = cloud->header.frame_id;
      marker.header.stamp = cloud_msg->header.stamp;
      marker.ns = "clusters";
      marker.id = cluster_id++;
      marker.type = visualization_msgs::Marker::LINE_LIST;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.orientation.w = 1.0;
      
      geometry_msgs::Point p[8];
      p[0].x = min_point[0]; p[0].y = min_point[1]; p[0].z = min_point[2];
      p[1].x = max_point[0]; p[1].y = min_point[1]; p[1].z = min_point[2];
      p[2].x = max_point[0]; p[2].y = max_point[1]; p[2].z = min_point[2];
      p[3].x = min_point[0]; p[3].y = max_point[1]; p[3].z = min_point[2];
      p[4].x = min_point[0]; p[4].y = min_point[1]; p[4].z = max_point[2];
      p[5].x = max_point[0]; p[5].y = min_point[1]; p[5].z = max_point[2];
      p[6].x = max_point[0]; p[6].y = max_point[1]; p[6].z = max_point[2];
      p[7].x = min_point[0]; p[7].y = max_point[1]; p[7].z = max_point[2];

      // Define the line segments of the bounding box
      for (int i = 0; i < 4; ++i)
      {
        marker.points.push_back(p[i]);
        marker.points.push_back(p[(i + 1) % 4]);
        marker.points.push_back(p[i + 4]);
        marker.points.push_back(p[(i + 1) % 4 + 4]);
        marker.points.push_back(p[i]);
        marker.points.push_back(p[i + 4]);
      }

      marker.scale.x = 0.1;
      marker.color.r = r / 255.0;
      marker.color.g = g / 255.0;
      marker.color.b = b / 255.0;
      marker.color.a = 1.0;

      marker_array.markers.push_back(marker);
    }

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*clustered_cloud, output);
    pub_.publish(output);

    marker_pub_.publish(marker_array);

    // bag_.write("clustered_cloud", ros::Time::now(), output);
  }

private:
  ros::NodeHandle nh_;
  ros::Subscriber sub_;
  ros::Publisher pub_;
  ros::Publisher marker_pub_;
  ros::Publisher debug_pub_;
  // rosbag::Bag bag_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  double ground_min_z_;
  double ground_max_z_;

  // Function to publish intermediate point clouds for debugging
  void publishDebugCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& label)
  {
    sensor_msgs::PointCloud2 debug_output;
    pcl::toROSMsg(*cloud, debug_output);
    debug_output.header.frame_id = cloud->header.frame_id;
    debug_output.header.stamp = ros::Time::now();
    debug_pub_.publish(debug_output);
    ROS_INFO("Published debug cloud: %s, size: %lu", label.c_str(), cloud->points.size());
  }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "point_cloud_clustering");
  PointCloudClustering pc_clustering;
  ros::spin();
  return 0;
}
