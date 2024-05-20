#ifndef POINT_CLOUD_CLUSTERING_HPP
#define POINT_CLOUD_CLUSTERING_HPP

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/conversions.h>
#include <visualization_msgs/MarkerArray.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/TransformStamped.h>
#include <random>
#include <vector>
#include <algorithm>
#include <limits>

class PointCloudClustering {
    public:
        PointCloudClustering();
        ~PointCloudClustering();
        void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg);
    
    private:
        ros::NodeHandle nh_;
        ros::Subscriber sub_;
        ros::Publisher pub_;
        ros::Publisher marker_pub_;
        ros::Publisher debug_pub_;
        tf2_ros::Buffer tf_buffer_;
        tf2_ros::TransformListener tf_listener_;

        double voxel_leaf_size_;
        double cropbox_min_x_, cropbox_min_y_, cropbox_min_z_;
        double cropbox_max_x_, cropbox_max_y_, cropbox_max_z_;
        double ground_min_z_, ground_max_z_;
        double sac_max_iterations_, sac_distance_threshold_;
        double sor_mean_k_, sor_stddev_mul_thresh_;
        double cluster_tolerance_;
        int min_cluster_size_, max_cluster_size_;

        int next_cluster_id_;
        std::vector<Eigen::Vector4f> previous_centroids_;
        std::vector<int> previous_cluster_ids_;

        pcl::PointCloud<pcl::PointXYZ>::Ptr downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr applyCropBox(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr removeGroundPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr removeOutliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
        std::vector<pcl::PointIndices> performClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
        void publishClusters(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<pcl::PointIndices>& cluster_indices, const std::string& frame_id, const ros::Time& timestamp);
        void trackClusters(const std::vector<Eigen::Vector4f>& current_centroids);

        void publishDebugCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& label);
        std::vector<int> hungarianAlgorithm(const std::vector<std::vector<double>>& cost_matrix);

};

#endif //POINT_CLOUD_CLUSTERING_HPP