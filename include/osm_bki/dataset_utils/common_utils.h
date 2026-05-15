#pragma once

#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "bkioctomap.h"
#include "markerarray_pub.h"
#include "osm_geometry.h"

// ---------------------------------------------------------------------------
// Common taxonomy (9 classes) — loaded from labels_common.yaml at runtime.
// ---------------------------------------------------------------------------
static constexpr int N_COMMON = 9;

struct MulticlassResult {
  pcl::PointCloud<pcl::PointXYZL>::Ptr cloud;
  std::vector<float> variances;
  /// Per-point probability distribution for soft counting (one row per point).
  /// If common_label_config_loaded_: size N_COMMON (mapped from network softmax).
  /// Else: size n_classes (raw network softmax). Set config num_class = n_classes when not using common.
  std::vector<std::vector<float>> common_probs;
  int n_classes = 0;
};

/// Voxel key for semantic uncertainty accumulation (input observation uncertainty per voxel)
struct VoxelKey {
  int x = 0, y = 0, z = 0;
  bool operator<(const VoxelKey& o) const {
    return std::tie(x, y, z) < std::tie(o.x, o.y, o.z);
  }
};

/// Parsed pose-trajectory data produced by dataset-specific pose readers
/// (see mcd_utils.h, kitti360_utils.h) and consumed by CommonUtils::set_pose_data.
struct PoseData {
  std::vector<Eigen::Matrix4d> lidar_poses;
  /// Maps pose index to the lidar/label file number on disk.
  std::vector<int> scan_indices;
  /// First pose before normalization; needed to align OSM geometry to the trajectory.
  Eigen::Matrix4d original_first_pose = Eigen::Matrix4d::Identity();
};

class CommonUtils {
 public:
  CommonUtils(rclcpp::Node::SharedPtr node,
               double resolution, double block_depth,
               double sf2, double ell,
               int num_class, double free_thresh,
               double occupied_thresh, float var_thresh,
               double ds_resolution,
               double free_resolution, double max_range,
               std::string map_topic,
               float prior);

  // --- Pose data ------------------------------------------------------------
  // Dataset-specific pose readers (osm_bki::mcd::read_lidar_poses,
  // osm_bki::kitti360::read_lidar_poses) parse files into a PoseData struct;
  // the node passes the result here.
  void set_pose_data(PoseData data);

  /// Get the original first pose (before transformation to origin). Needed to align OSM data with the same coordinate frame.
  Eigen::Matrix4d getOriginalFirstPose() const;

  // --- Multiclass / label configuration -------------------------------------
  /// Enable multiclass for inferred labels: read per-class confidence scores and take argmax.
  void set_inferred_multiclass_mode(bool use_mc, const std::string& multiclass_dir);

  /// Enable multiclass for GT labels (e.g. one-hot or per-class scores). Uses gt_label_dir.
  void set_gt_multiclass_mode(bool use_mc);

  /// Load learning_map_inv from a label config YAML (e.g. labels_semkitti.yaml) for inferred labels.
  bool load_label_config(const std::string& yaml_path);

  /// Load learning_map_inv for GT labels (when GT is multiclass from a model).
  bool load_gt_label_config(const std::string& yaml_path);

  /// Load common taxonomy mappings from labels_common.yaml.
  /// @param yaml_path       Path to labels_common.yaml.
  /// @param inferred_key    "mcd", "semkitti", or "kitti360" — picks <key>_to_common for inferred labels.
  /// @param gt_key          "mcd", "semkitti", or "kitti360" — picks <key>_to_common for GT labels.
  bool load_common_label_config(const std::string& yaml_path,
                                const std::string& inferred_key,
                                const std::string& gt_key);

  // --- Uncertainty / confusion matrix --------------------------------------
  void set_uncertainty_filter(bool enabled, const std::string& labels_key,
                              const std::string& mode = "confusion_matrix",
                              float drop_percent = 10.0f,
                              float min_weight = 0.1f);

  /// Load a pre-computed confusion matrix from YAML (rows=predicted, cols=true).
  /// Computes per-class precision and stores it for filtering.
  bool load_confusion_matrix(const std::string& yaml_path);

  // --- Visualization configuration -----------------------------------------
  /// Set map visualization color mode: semantic class or OSM prior (building/road/grassland/tree).
  void set_color_mode(osm_bki::MapColorMode mode);

  /// Enable variance visualization on a separate topic. Jet colormap: blue=low variance, red=high.
  void set_publish_variance(bool enabled, const std::string& topic);

  /// Enable semantic uncertainty (input observation) map visualization. Voxel map with jet colormap: blue=confident, red=uncertain.
  /// Same pattern as variance map. Requires inferred_use_multiclass.
  void set_publish_semantic_uncertainty(bool enabled, const std::string& topic);

  // --- OSM geometry pass-through setters -----------------------------------
  void set_osm_buildings(const std::vector<osm_bki::Geometry2D>& buildings);
  void set_osm_roads(const std::vector<osm_bki::Geometry2D>& roads);
  void set_osm_sidewalks(const std::vector<osm_bki::Geometry2D>& sidewalks);
  void set_osm_cycleways(const std::vector<osm_bki::Geometry2D>& cycleways);
  void set_osm_grasslands(const std::vector<osm_bki::Geometry2D>& grasslands);
  void set_osm_trees(const std::vector<osm_bki::Geometry2D>& trees);
  void set_osm_forests(const std::vector<osm_bki::Geometry2D>& forests);
  void set_osm_tree_points(const std::vector<std::pair<float, float>>& tree_points);
  void set_osm_tree_point_radius(float radius_m);
  void set_osm_road_width(float width_m);
  void set_osm_sidewalk_width(float width_m);
  void set_osm_cycleway_width(float width_m);
  void set_osm_fence_width(float width_m);
  void set_osm_parking(const std::vector<osm_bki::Geometry2D>& parking);
  void set_osm_fences(const std::vector<osm_bki::Geometry2D>& fences);
  void set_osm_decay_meters(float decay_m);
  void set_osm_prior_strength(float strength);
  void set_osm_dirichlet_prior_strength(float strength);
  void set_osm_scan_radius_extension(float factor);

  void set_height_kernel_params(float lambda,
                                const std::vector<float>& mu,
                                const std::vector<float>& tau,
                                const std::vector<float>& dead_zone,
                                bool redistribute,
                                float gate,
                                float sensor_mounting_height);

  /// Enable OSM prior map on a separate topic. Renders as a 2D raster (single z layer)
  /// over the BKI's xy bounding box; OSM priors computed on-the-fly (no height filter).
  void set_publish_osm_prior_map(bool enabled, const std::string& topic,
                                 osm_bki::MapColorMode osm_color_mode, float raster_z = 0.f);

  /// Enable OSM-converted map on a separate topic. Renders the post-CM, post-height-filter
  /// argmax common-class prediction at every occupied voxel; uses semantic colors.
  void set_publish_osm_converted_map(bool enabled, const std::string& topic);

  /// Load optional geometry parameters used by OSM priors/visualization.
  /// Expected node:
  ///   osm_geometry_parameters:
  ///     road_width_meters: ...
  ///     sidewalk_width_meters: ...
  ///     cycleway_width_meters: ...
  ///     fence_width_meters: ...
  ///     tree_point_radius_meters: ...
  ///     osm_decay_meters: ...
  bool load_osm_geometry_parameters(const std::string& yaml_path);

  bool load_osm_confusion_matrix(const std::string& yaml_path);

  // --- Scan processing ------------------------------------------------------
  /// Return true if both the lidar bin and label/multiclass file exist for the given scan file number.
  bool scan_and_label_exist(const std::string& input_data_dir, const std::string& input_label_dir, int scan_file_num);

  bool process_scans(std::string input_data_dir, std::string input_label_dir, int scan_num, double keyframe_dist, bool query, bool publish_semantic_occ_map);

  bool any_publish_enabled() const;

  void publish_map();

  // --- Colors / calibration / evaluation ------------------------------------
  bool load_colors_from_params();
  bool load_colors_from_yaml(const std::string& yaml_file_path);
  void set_up_evaluation(const std::string gt_label_dir, const std::string evaluation_result_dir);
  bool load_calibration_from_params();
  bool load_calibration_from_yaml(const std::string& yaml_file);

  void query_scan(std::string input_data_dir, std::string input_label_dir, int pose_idx);

 private:
  bool load_learning_map_inv_(const std::string& yaml_path,
                              std::map<int, int>& out_map,
                              const std::string& label);

  pcl::PointCloud<pcl::PointXYZL>::Ptr read_scan_with_labels(std::string fn, std::string fn_label, bool use_gt_mapping = false);

  /// Read lidar scan + multiclass confidence scores (float16), take argmax,
  /// apply learning_map_inv.  Also computes per-point variance of the class
  /// probability distribution (used for uncertainty filtering).
  MulticlassResult read_scan_with_multiclass_scores(const std::string& fn, const std::string& fn_multiclass);

  /// Read GT from multiclass/one-hot format (float16, n_points * n_classes).
  /// Takes argmax, applies gt_learning_map_inv, then gt_to_common. Returns cloud with labels in common taxonomy.
  pcl::PointCloud<pcl::PointXYZL>::Ptr read_gt_scan_multiclass(const std::string& fn_scan, const std::string& fn_gt_multiclass);

  /// Read GT label file (uint32 per point). Returns empty vector on failure.
  std::vector<uint32_t> read_gt_labels(const std::string& dir, int scan_file_num);

  // --- Members --------------------------------------------------------------
  rclcpp::Node::SharedPtr node_;
  double resolution_;
  int num_class_;
  double ds_resolution_;
  double free_resolution_;
  double max_range_;
  osm_bki::SemanticBKIOctoMap* map_;
  osm_bki::MarkerArrayPub* m_pub_;
  bool publish_semantic_occ_map_{true};
  osm_bki::MarkerArrayPub* variance_pub_{nullptr};
  bool publish_variance_{false};
  osm_bki::MarkerArrayPub* osm_prior_map_pub_{nullptr};
  bool publish_osm_prior_map_{false};
  osm_bki::MapColorMode osm_prior_map_color_mode_{osm_bki::MapColorMode::OSMBlend};
  float osm_prior_map_z_{0.f};  // ground-plane z for the 2D raster
  osm_bki::MarkerArrayPub* osm_converted_map_pub_{nullptr};
  bool publish_osm_converted_map_{false};
  std::string colors_file_path_;  // last-loaded semantic colors yaml; reused for downstream pubs

  // Per-scan xy bbox in the map frame, captured from the most recently inserted
  // cloud. Used as the raster footprint for the OSM prior-map visualization so
  // it doesn't grow with the cumulative map.
  bool scan_xy_set_{false};
  float scan_xy_min_x_{0.f};
  float scan_xy_min_y_{0.f};
  float scan_xy_max_x_{0.f};
  float scan_xy_max_y_{0.f};
  std::string variance_topic_;
  osm_bki::MarkerArrayPub* semantic_uncertainty_pub_{nullptr};
  bool publish_semantic_uncertainty_{false};
  std::string semantic_uncertainty_topic_;
  std::map<VoxelKey, std::pair<float, int>> semantic_uncertainty_acc_;  // per-voxel: (sum_uncertainty, count)
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;  // Publisher for individual scan point clouds
  tf2_ros::TransformBroadcaster tf_broadcaster_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::ofstream pose_file_;
  std::vector<Eigen::Matrix4d> lidar_poses_;
  std::vector<int> scan_indices_;  // Maps pose index to actual scan file number (from CSV "num" column)
  std::string gt_label_dir_;
  std::string evaluation_result_dir_;
  Eigen::Matrix4d init_trans_to_ground_;
  Eigen::Matrix4d body_to_lidar_tf_;
  Eigen::Matrix4d original_first_pose_;

  // Multiclass settings (inferred and GT)
  bool inferred_use_multiclass_;
  std::string inferred_multiclass_dir_;
  std::map<int, int> learning_map_inv_;   // for inferred: model output index → raw label
  bool gt_use_multiclass_;
  std::map<int, int> gt_learning_map_inv_;  // for GT: model output index → raw label

  // Common taxonomy mappings (loaded from labels_common.yaml)
  bool common_label_config_loaded_;
  std::map<int, int> inferred_to_common_;  // raw inferred label → common class index
  std::map<int, int> gt_to_common_;        // raw GT label → common class index

  // Uncertainty filtering
  bool use_uncertainty_filter_;
  bool confusion_matrix_loaded_;
  std::string inferred_labels_key_;  // "mcd", "semkitti", or "kitti360"
  std::string uncertainty_filter_mode_;  // "confusion_matrix" or "top_percent"
  float uncertainty_drop_percent_;       // for top_percent mode: discount this % of most uncertain points
  float uncertainty_min_weight_;         // minimum kernel weight for the most uncertain points
  int confusion_matrix_[N_COMMON][N_COMMON];
  float class_precision_[N_COMMON];
  int total_points_processed_;
  int total_points_filtered_;
  std::vector<float> scan_point_weights_;
  /// Per-point common-class probabilities for soft counting (same size as cloud when multiclass + common config).
  std::vector<std::vector<float>> scan_common_probs_;
};
