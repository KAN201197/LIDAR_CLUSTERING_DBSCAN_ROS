# Lidar Clustering with DBSCAN in ROS
This ROS package performs point cloud clustering using DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm. It segments the ground plane, removes outliers, and then clusters the remaining points. The results are visualized using RViz markers.

## Overview
This package subscribes to a point cloud topic, applies a pass-through filter to limit the height of points, segments the ground plane using RANSAC, removes outliers, and performs DBSCAN clustering on the remaining points. The clustered points are published as a new point cloud, and bounding boxes are visualized using RViz markers.

## Installation
1. Build ROS workspace and clone this repository to ROS workspace

       mkdir ~catkin_ws/src -p

       cd catkin_ws/src

       git clone https://github.com/KAN201197/LIDAR_CLUSTERING_DBSCAN_ROS.git

2. Navigate to ROS workspace, build the package, and source the workspace

       cd catkin_ws

       catkin_make

       source devel/setup.bash

## Usage
1. Run the node

       roslaunch point_cloud_clustering point_cloud_clustering.launch

2. Visualize in the RVIZ
   
   - Add the clustered_cloud topic to visualize the clustered points.
   - Add the cluster_markers topic to visualize the bounding boxes of the clusters.

## Parameters
The node has several parameters that can be adjusted to fit specific use cases:

- **ground_min_z** (double, default: -2.0): Minimum Z value for pass-through filter.
- **ground_max_z** (double, default: 3.0): Maximum Z value for pass-through filter.
- **ransac_max_iterations** (int, default: 10000): Maximum number of iterations for RANSAC algorithm.
- **ransac_distance_threshold** (double, default: 0.1): Distance threshold for RANSAC to consider a point as an inlier.
- **cluster_tolerance** (double, default: 0.3): The spatial distance tolerance for clustering.
- **min_cluster_size** (int, default: 1000): Minimum number of points that a cluster needs to be considered valid.
- **max_cluster_size** (int, default: 10000): Maximum number of points that a cluster can have.

## Topic

### Subscribed Topic
- **/me5413/lidar_top** (sensor_msgs/PointCloud2): Input point cloud from the LiDAR sensor.

### Published Topics
- **clustered_cloud** (sensor_msgs/PointCloud2): Output point cloud with clustered points colored based on their cluster ID.
- **cluster_markers** (visualization_msgs/MarkerArray): Markers for visualizing the bounding boxes of clusters in RViz.
- **debug_cloud** (sensor_msgs/PointCloud2): Intermediate point clouds for debugging purposes.    
