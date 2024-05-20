#include "point_cloud_clustering/point_cloud_clustering.hpp"

PointCloudClustering::PointCloudClustering()
    : tf_listener_(tf_buffer_)
{
    ros::NodeHandle pnh("~");
    if (!pnh.getParam("voxel_leaf_size", voxel_leaf_size_))
        ROS_ERROR("Failed to get parameter voxel_leaf_size");
    if (!pnh.getParam("cropbox_min_x", cropbox_min_x_))
        ROS_ERROR("Failed to get parameter cropbox_min_x");
    if (!pnh.getParam("cropbox_min_y", cropbox_min_y_))
        ROS_ERROR("Failed to get parameter cropbox_min_y");
    if (!pnh.getParam("cropbox_min_z", cropbox_min_z_))
        ROS_ERROR("Failed to get parameter cropbox_min_z");
    if (!pnh.getParam("cropbox_max_x", cropbox_max_x_))
        ROS_ERROR("Failed to get parameter cropbox_max_x");
    if (!pnh.getParam("cropbox_max_y", cropbox_max_y_))
        ROS_ERROR("Failed to get parameter cropbox_max_y");
    if (!pnh.getParam("cropbox_max_z", cropbox_max_z_))
        ROS_ERROR("Failed to get parameter cropbox_max_z");
    if (!pnh.getParam("ground_min_z", ground_min_z_))
        ROS_ERROR("Failed to get parameter ground_min_z");
    if (!pnh.getParam("ground_max_z", ground_max_z_))
        ROS_ERROR("Failed to get parameter ground_max_z");
    if (!pnh.getParam("cluster_tolerance", cluster_tolerance_))
        ROS_ERROR("Failed to get parameter cluster_tolerance");
    if (!pnh.getParam("min_cluster_size", min_cluster_size_))
        ROS_ERROR("Failed to get parameter min_cluster_size");
    if (!pnh.getParam("max_cluster_size", max_cluster_size_))
        ROS_ERROR("Failed to get parameter max_cluster_size");
    if (!pnh.getParam("sac_max_iterations", sac_max_iterations_))
        ROS_ERROR("Failed to get parameter sac_max_iterations");
    if (!pnh.getParam("sac_distance_threshold", sac_distance_threshold_))
        ROS_ERROR("Failed to get parameter sac_distance_threshold");
    if (!pnh.getParam("sor_mean_k", sor_mean_k_))
        ROS_ERROR("Failed to get parameter sor_mean_k");
    if (!pnh.getParam("sor_stddev_mul_thresh", sor_stddev_mul_thresh_))
        ROS_ERROR("Failed to get parameter sor_stddev_mul_thresh");

    sub_ = nh_.subscribe("/me5413/lidar_top", 1, &PointCloudClustering::pointCloudCallback, this);
    pub_ = nh_.advertise<sensor_msgs::PointCloud2>("clustered_cloud", 1);
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("cluster_markers", 1);
    debug_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("debug_cloud", 1);
}

PointCloudClustering::~PointCloudClustering()
{
}

pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudClustering::downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    ROS_INFO("Downsampling cloud...");
    
    if (!cloud)
    {
      ROS_ERROR("Input cloud is null");
      return nullptr;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr voxel_filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(cloud);
    voxel_filter.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
    voxel_filter.filter(*voxel_filtered_cloud);

    ROS_INFO("Downsampling complete. Resulting cloud size: %lu", voxel_filtered_cloud->points.size());

    return voxel_filtered_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudClustering::applyCropBox(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{   
    ROS_INFO("Applying crop box...");

    if (!cloud)
    {
      ROS_ERROR("Input cloud is null");
      return nullptr;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::CropBox<pcl::PointXYZ> crop_box;
    crop_box.setMin(Eigen::Vector4f(cropbox_min_x_, cropbox_min_y_, cropbox_min_z_, 1.0));
    crop_box.setMax(Eigen::Vector4f(cropbox_max_x_, cropbox_max_y_, cropbox_max_z_, 1.0));
    crop_box.setInputCloud(cloud);
    crop_box.filter(*cropped_cloud);

    ROS_INFO("Crop box applied. Resulting cloud size: %lu", cropped_cloud->points.size());

    return cropped_cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudClustering::removeGroundPlane(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    ROS_INFO("Applying ground remove");
    
    if (!cloud)
    {
      ROS_ERROR("Input cloud is null");
      return nullptr;
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(ground_min_z_, ground_max_z_);
    pass.filter(*filtered_cloud);

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr ground_inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(sac_max_iterations_);
    seg.setDistanceThreshold(sac_distance_threshold_);
    seg.setInputCloud(filtered_cloud);
    seg.segment(*ground_inliers, *coefficients);

    if (ground_inliers->indices.empty())
    {
        ROS_WARN("Could not estimate a planar model for the given dataset.");
        return filtered_cloud;
    }

    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(ground_inliers);
    extract.setNegative(true);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground(new pcl::PointCloud<pcl::PointXYZ>());
    extract.filter(*cloud_no_ground);

    ROS_INFO("Removing ground plane. Resulting cloud size: %lu", cloud_no_ground->points.size());

    return cloud_no_ground;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudClustering::removeOutliers(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    ROS_INFO("Applying removeOutliers");

    if (!cloud)
    {
      ROS_ERROR("Input cloud is null");
      return nullptr;
    }
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground_filtered(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(sor_mean_k_);
    sor.setStddevMulThresh(sor_stddev_mul_thresh_);
    sor.filter(*cloud_no_ground_filtered);

    ROS_INFO("Removing outliers. Resulting cloud size: %lu", cloud_no_ground_filtered->points.size());

    return cloud_no_ground_filtered;
}

std::vector<pcl::PointIndices> PointCloudClustering::performClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    ROS_INFO("Applying Clustering");

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(cloud);

    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_);
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    ROS_INFO("Clustering Finish. Cluster size: %lu", cluster_indices.size());

    return cluster_indices;
}

void PointCloudClustering::publishClusters(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<pcl::PointIndices>& cluster_indices, const std::string& frame_id, const ros::Time& timestamp)
{
    ROS_INFO("START PUBLISH CLUSTER");

    visualization_msgs::MarkerArray marker_array;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    for (const auto& indices : cluster_indices)
    {
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud, indices.indices, centroid);

        pcl::PointXYZRGB color_point;
        color_point.r = static_cast<uint8_t>(rand() % 256);
        color_point.g = static_cast<uint8_t>(rand() % 256);
        color_point.b = static_cast<uint8_t>(rand() % 256);

        for (const auto& idx : indices.indices)
        {
            pcl::PointXYZRGB point;
            point.x = cloud->points[idx].x;
            point.y = cloud->points[idx].y;
            point.z = cloud->points[idx].z;
            point.r = color_point.r;
            point.g = color_point.g;
            point.b = color_point.b;
            colored_cloud->points.push_back(point);
        }

        // Create bounding box marker
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cloud, indices.indices, min_pt, max_pt);

        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = timestamp;
        marker.ns = "clusters";
        marker.id = next_cluster_id_++;
        marker.type = visualization_msgs::Marker::CUBE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = (min_pt.x() + max_pt.x()) / 2.0;
        marker.pose.position.y = (min_pt.y() + max_pt.y()) / 2.0;
        marker.pose.position.z = (min_pt.z() + max_pt.z()) / 2.0;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = max_pt.x() - min_pt.x();
        marker.scale.y = max_pt.y() - min_pt.y();
        marker.scale.z = max_pt.z() - min_pt.z();
        marker.color.r = static_cast<float>(color_point.r) / 255.0;
        marker.color.g = static_cast<float>(color_point.g) / 255.0;
        marker.color.b = static_cast<float>(color_point.b) / 255.0;
        marker.color.a = 0.3;
        marker.lifetime = ros::Duration(0.1);

        marker_array.markers.push_back(marker);
    }

    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*colored_cloud, output);
    output.header.frame_id = frame_id;
    output.header.stamp = timestamp;
    pub_.publish(output);
    marker_pub_.publish(marker_array);

    ROS_INFO("FINISH PUBLISH CLUSTER");
}

void PointCloudClustering::trackClusters(const std::vector<Eigen::Vector4f>& current_centroids)
{
    if (previous_centroids_.empty() || current_centroids.empty())
    {
        ROS_ERROR("Empty centroids detected.");
        return;
    }

    if (previous_centroids_.size() != previous_cluster_ids_.size())
    {
        ROS_ERROR("Size mismatch between previous centroids and cluster ids.");
        return;
    }

    size_t max_size = std::max(previous_centroids_.size(), current_centroids.size());

    std::vector<std::vector<double>> cost_matrix(max_size, std::vector<double>(max_size, std::numeric_limits<double>::infinity()));
    for (size_t i = 0; i < previous_centroids_.size(); ++i)
    {
        for (size_t j = 0; j < current_centroids.size(); ++j)
        {
            if (!std::isfinite(previous_centroids_[i].norm()) || !std::isfinite(current_centroids[j].norm()))
            {
                ROS_ERROR("Invalid centroid norm detected.");
                return;
            }
            cost_matrix[i][j] = (previous_centroids_[i] - current_centroids[j]).norm();
        }
    }

    ROS_INFO("Cost matrix generated.");

    std::vector<int> assignment = hungarianAlgorithm(cost_matrix);

    ROS_INFO("Hungarian algorithm completed.");

    if (assignment.size() != current_centroids.size())
    {
        ROS_ERROR("Size mismatch between assignment and current centroids.");
        return;
    }

    std::vector<int> new_cluster_ids(current_centroids.size());
    for (size_t i = 0; i < assignment.size(); ++i)
    {
        if (assignment[i] != -1)
        {
            new_cluster_ids[assignment[i]] = previous_cluster_ids_[i];
        }
        else
        {
            new_cluster_ids[assignment[i]] = next_cluster_id_++;
        }
    }

    ROS_INFO("Cluster IDs updated.");

    previous_centroids_ = current_centroids;
    previous_cluster_ids_ = new_cluster_ids;

    ROS_INFO("Previous centroids and cluster IDs updated.");
}

void PointCloudClustering::publishDebugCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::string& label)
{
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*cloud, output);
    output.header.frame_id = cloud->header.frame_id;
    output.header.stamp = ros::Time::now();
    debug_pub_.publish(output);
    ROS_INFO("%s size: %lu", label.c_str(), cloud->points.size());
}

std::vector<int> PointCloudClustering::hungarianAlgorithm(const std::vector<std::vector<double>>& cost_matrix)
{
    int n = cost_matrix.size();
    int m = cost_matrix[0].size();

    if (n == 0 || m == 0 || n != m)
    {
        ROS_ERROR("Invalid cost matrix dimensions");
        return std::vector<int>();
    }

    ROS_INFO("Starting Hungarian algorithm...");
    ROS_INFO("n = %d, m = %d", n, m);

    // Normalize cost matrix values
    double max_cost = std::numeric_limits<double>::min();
    double min_cost = std::numeric_limits<double>::max();
    
    for (const auto& row : cost_matrix) {
        for (double cost : row) {
            if (cost > max_cost) max_cost = cost;
            if (cost < min_cost) min_cost = cost;
        }
    }
    
    std::vector<std::vector<double>> normalized_cost_matrix = cost_matrix;
    ROS_INFO("Normalized cost matrix:");
    for (auto& row : normalized_cost_matrix) {
        std::ostringstream oss;
        for (double& cost : row) {
            cost = (cost - min_cost) / (max_cost - min_cost); // Normalize between 0 and 1
            oss << cost << " ";
        }
        ROS_INFO("%s", oss.str().c_str());
    }

    std::vector<int> assignment(n, -1);
    std::vector<double> u(n, 0), v(m, 0);
    std::vector<int> p(m), way(m, -1);
    std::vector<double> minv(m);
    std::vector<bool> used(m);

    for (int i = 0; i < n; ++i)
    {
        std::fill(minv.begin(), minv.end(), std::numeric_limits<double>::max());
        std::fill(used.begin(), used.end(), false);
        int j0 = 0;
        p[0] = i;
        int j1 = 0;

        ROS_INFO("Processing row %d", i);

        // Debugging initial values of minv and used
        ROS_INFO("Initial minv values for row %d:", i);
        for (const auto& val : minv) {
            ROS_INFO("%f", val);
        }

        ROS_INFO("Initial used values for row %d:", i);
        for (const auto& val : used) {
            ROS_INFO("%d", val);
        }

        while (true)
        {
            j1 = -1;
            for (int j = 0; j < m; ++j)
            {
                if (!used[j])
                {
                    double cur = normalized_cost_matrix[p[j0]][j] - u[p[j0]] - v[j];
                    if (cur < minv[j])
                    {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (j1 == -1 || minv[j] < minv[j1])
                    {
                        j1 = j;
                    }
                }
            }
            ROS_INFO("minv: %f, j1: %d", (j1 != -1 ? minv[j1] : 0.0), j1);

            if (j1 == -1)
            {
                ROS_ERROR("No valid column found for row %d", i);
                ROS_INFO("minv values: ");
                for (int j = 0; j < m; ++j)
                {
                    ROS_INFO("%f ", minv[j]);
                }
                return std::vector<int>(); // Error handling
            }

            double delta = minv[j1];
            for (int j = 0; j < m; ++j)
            {
                if (used[j])
                {
                    u[p[j]] += delta;
                    v[j] -= delta;
                }
                else
                {
                    minv[j] -= delta;
                }
            }
            u[i] += delta;
            used[j1] = true;
            j0 = way[j1];
            if (p[j1] == -1)
                break;
            else
                j1 = way[j1];
        }

        // Debugging values after processing each row
        ROS_INFO("Values after processing row %d:", i);
        ROS_INFO("u values:");
        for (const auto& val : u) {
            ROS_INFO("%f", val);
        }

        ROS_INFO("v values:");
        for (const auto& val : v) {
            ROS_INFO("%f", val);
        }

        ROS_INFO("used values:");
        for (const auto& val : used) {
            ROS_INFO("%d", val);
        }

        while (j1 != -1)
        {
            int j0 = way[j1];
            p[j1] = p[j0];
            j1 = j0;
        }
    }

    for (int j = 0; j < m; ++j)
    {
        if (p[j] != -1)
        {
            assignment[p[j]] = j;
        }
    }

    ROS_INFO("Hungarian algorithm completed. Assignment size: %zu", assignment.size());
    return assignment;
}

void PointCloudClustering::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    auto downsampled_cloud = downsample(cloud);
    auto cropped_cloud = applyCropBox(downsampled_cloud);
    auto cloud_no_ground = removeGroundPlane(cropped_cloud);
    auto cloud_no_ground_filtered = removeOutliers(cloud_no_ground);
    publishDebugCloud(cloud_no_ground, "downsampling_cloud");
    auto cluster_indices = performClustering(cloud_no_ground_filtered);

    std::vector<Eigen::Vector4f> current_centroids;
    for (const auto& indices : cluster_indices)
    {
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*cloud_no_ground_filtered, indices.indices, centroid);
        current_centroids.push_back(centroid);
    }

    trackClusters(current_centroids);
    publishClusters(cloud_no_ground_filtered, cluster_indices, cloud_msg->header.frame_id, cloud_msg->header.stamp);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "point_cloud_clustering_node");
    PointCloudClustering point_cloud_clustering;

    ros::spin();

    return 0;
}
