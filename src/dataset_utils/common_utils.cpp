#include "common_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <thread>

#include <yaml-cpp/yaml.h>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>

namespace {

/// Convert IEEE 754 half-precision (uint16) to single-precision float.
inline float half_to_float(uint16_t h) {
    uint32_t sign     = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa = h & 0x03FFu;

    if (exponent == 0) {
        if (mantissa == 0) {
            float f; uint32_t r = sign;
            std::memcpy(&f, &r, sizeof(f));
            return f;
        }
        while (!(mantissa & 0x0400u)) { mantissa <<= 1; exponent--; }
        exponent++; mantissa &= ~0x0400u;
    } else if (exponent == 31) {
        uint32_t r = sign | 0x7F800000u | (mantissa << 13);
        float f; std::memcpy(&f, &r, sizeof(f));
        return f;
    }

    exponent += (127 - 15);
    uint32_t r = sign | (exponent << 23) | (mantissa << 13);
    float f; std::memcpy(&f, &r, sizeof(f));
    return f;
}

}  // namespace

CommonUtils::CommonUtils(rclcpp::Node::SharedPtr node,
                 double resolution, double block_depth,
                 double sf2, double ell,
                 int num_class, double free_thresh,
                 double occupied_thresh, float var_thresh,
                 double ds_resolution,
                 double free_resolution, double max_range,
                 std::string map_topic,
                 float prior)
  : node_(node)
  , resolution_(resolution)
  , num_class_(num_class)
  , ds_resolution_(ds_resolution)
  , free_resolution_(free_resolution)
  , max_range_(max_range)
  , tf_broadcaster_(node)
  , tf_buffer_(std::make_shared<tf2_ros::Buffer>(node->get_clock()))
  , tf_listener_(*tf_buffer_) {
  if (!node_) {
    RCLCPP_WARN_STREAM(rclcpp::get_logger("common_utils"), "WARNING: CommonUtils constructor: node_ is null!");
  }
  RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: CommonUtils constructor: Creating octomap");
  map_ = new osm_bki::SemanticBKIOctoMap(resolution, block_depth, num_class, sf2, ell, prior, var_thresh, free_thresh, occupied_thresh);
  if (!map_) {
    RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Failed to create SemanticBKIOctoMap!");
  } else {
    RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Octomap created successfully");
  }

  RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Creating MarkerArrayPub");
  m_pub_ = new osm_bki::MarkerArrayPub(node_, map_topic, resolution);
  if (!m_pub_) {
    RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Failed to create MarkerArrayPub!");
  } else {
    RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: MarkerArrayPub created successfully");
  }

  // Publisher for individual scan point clouds
  RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Creating pointcloud publisher");
  pointcloud_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/mcd_scan_pointcloud", 10);
  if (!pointcloud_pub_) {
    RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Failed to create pointcloud publisher!");
  } else {
    RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Pointcloud publisher created successfully");
  }

  // Identity transformation for MCD (poses are already in world frame)
  init_trans_to_ground_ = Eigen::Matrix4d::Identity();
  // Body to LiDAR transformation (must be loaded from calibration - no default identity)
  // Will be set by load_calibration_from_params() - if not set, will error
  body_to_lidar_tf_ = Eigen::Matrix4d::Zero();  // Set to zero to detect if not loaded
  original_first_pose_ = Eigen::Matrix4d::Identity();  // Will be set when poses are loaded
  scan_indices_.clear();
  inferred_use_multiclass_ = false;
  gt_use_multiclass_ = false;
  common_label_config_loaded_ = false;
  use_uncertainty_filter_ = false;
  confusion_matrix_loaded_ = false;
  uncertainty_filter_mode_ = "confusion_matrix";
  uncertainty_drop_percent_ = 10.0f;
  uncertainty_min_weight_ = 0.1f;
  total_points_processed_ = 0;
  total_points_filtered_ = 0;
  std::memset(confusion_matrix_, 0, sizeof(confusion_matrix_));
  std::memset(class_precision_, 0, sizeof(class_precision_));
  RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: CommonUtils constructor completed");
}

void CommonUtils::set_pose_data(PoseData data) {
  lidar_poses_ = std::move(data.lidar_poses);
  scan_indices_ = std::move(data.scan_indices);
  original_first_pose_ = data.original_first_pose;
}

Eigen::Matrix4d CommonUtils::getOriginalFirstPose() const {
  return original_first_pose_;
}

void CommonUtils::set_inferred_multiclass_mode(bool use_mc, const std::string& multiclass_dir) {
  inferred_use_multiclass_ = use_mc;
  inferred_multiclass_dir_ = multiclass_dir;
  if (use_mc) {
    RCLCPP_INFO_STREAM(node_->get_logger(),
        "Inferred labels multiclass mode enabled. Scores dir: " << multiclass_dir);
  }
}

void CommonUtils::set_gt_multiclass_mode(bool use_mc) {
  gt_use_multiclass_ = use_mc;
  if (use_mc) {
    RCLCPP_INFO_STREAM(node_->get_logger(), "GT labels multiclass mode enabled.");
  }
}

bool CommonUtils::load_label_config(const std::string& yaml_path) {
  return load_learning_map_inv_(yaml_path, learning_map_inv_, "inferred");
}

bool CommonUtils::load_gt_label_config(const std::string& yaml_path) {
  return load_learning_map_inv_(yaml_path, gt_learning_map_inv_, "GT");
}

bool CommonUtils::load_learning_map_inv_(const std::string& yaml_path,
                                     std::map<int, int>& out_map,
                                     const std::string& label) {
  try {
    YAML::Node cfg = YAML::LoadFile(yaml_path);
    if (!cfg["learning_map_inv"]) {
      RCLCPP_ERROR_STREAM(node_->get_logger(), "No 'learning_map_inv' key in " << yaml_path);
      return false;
    }
    out_map.clear();
    for (auto it = cfg["learning_map_inv"].begin();
         it != cfg["learning_map_inv"].end(); ++it) {
      int class_idx = it->first.as<int>();
      int label_id  = it->second.as<int>();
      out_map[class_idx] = label_id;
    }
    RCLCPP_INFO_STREAM(node_->get_logger(),
        "Loaded " << label << " learning_map_inv with " << out_map.size()
        << " entries from " << yaml_path);
    return true;
  } catch (const std::exception& e) {
    RCLCPP_ERROR_STREAM(node_->get_logger(),
        "Failed to load label config: " << e.what());
    return false;
  }
}

bool CommonUtils::load_common_label_config(const std::string& yaml_path,
                                       const std::string& inferred_key,
                                       const std::string& gt_key) {
  try {
    YAML::Node cfg = YAML::LoadFile(yaml_path);
    std::string inf_map_key = inferred_key + "_to_common";
    std::string gt_map_key  = gt_key + "_to_common";

    if (!cfg[inf_map_key]) {
      RCLCPP_ERROR_STREAM(node_->get_logger(),
          "No '" << inf_map_key << "' key in " << yaml_path);
      return false;
    }
    if (!cfg[gt_map_key]) {
      RCLCPP_ERROR_STREAM(node_->get_logger(),
          "No '" << gt_map_key << "' key in " << yaml_path);
      return false;
    }

    inferred_to_common_.clear();
    for (auto it = cfg[inf_map_key].begin(); it != cfg[inf_map_key].end(); ++it)
      inferred_to_common_[it->first.as<int>()] = it->second.as<int>();

    gt_to_common_.clear();
    for (auto it = cfg[gt_map_key].begin(); it != cfg[gt_map_key].end(); ++it)
      gt_to_common_[it->first.as<int>()] = it->second.as<int>();

    common_label_config_loaded_ = true;
    RCLCPP_INFO_STREAM(node_->get_logger(),
        "Loaded common label config from " << yaml_path
        << ": inferred mapping '" << inf_map_key << "' (" << inferred_to_common_.size()
        << " entries), GT mapping '" << gt_map_key << "' (" << gt_to_common_.size()
        << " entries)");
    return true;
  } catch (const std::exception& e) {
    RCLCPP_ERROR_STREAM(node_->get_logger(), "Failed to load common label config: " << e.what());
    return false;
  }
}

void CommonUtils::set_uncertainty_filter(bool enabled, const std::string& labels_key,
                                     const std::string& mode,
                                     float drop_percent,
                                     float min_weight) {
  use_uncertainty_filter_ = enabled;
  inferred_labels_key_ = labels_key;
  uncertainty_filter_mode_ = mode;
  uncertainty_drop_percent_ = drop_percent;
  uncertainty_min_weight_ = min_weight;
  if (enabled) {
    RCLCPP_INFO_STREAM(node_->get_logger(),
        "Uncertainty filtering enabled (mode=" << mode
        << ", labels_key=" << labels_key
        << (mode == "top_percent"
            ? ", drop_percent=" + std::to_string(drop_percent)
              + ", min_weight=" + std::to_string(min_weight)
            : "")
        << ")");
  }
}

bool CommonUtils::load_confusion_matrix(const std::string& yaml_path) {
  try {
    YAML::Node root = YAML::LoadFile(yaml_path);
    if (!root["confusion_matrix"]) {
      RCLCPP_ERROR_STREAM(node_->get_logger(), "No 'confusion_matrix' key in " << yaml_path);
      return false;
    }
    std::memset(confusion_matrix_, 0, sizeof(confusion_matrix_));
    auto cm = root["confusion_matrix"];
    for (auto it = cm.begin(); it != cm.end(); ++it) {
      int pred_cls = it->first.as<int>();
      if (pred_cls < 0 || pred_cls >= N_COMMON) continue;
      auto row = it->second;
      // Columns are classes 1..(N_COMMON-1), stored as a sequence of (N_COMMON-1) values
      if (static_cast<int>(row.size()) != N_COMMON - 1) {
        RCLCPP_WARN_STREAM(node_->get_logger(),
            "Confusion matrix row " << pred_cls << " has " << row.size()
            << " columns (expected " << (N_COMMON - 1) << "), skipping");
        continue;
      }
      for (int c = 0; c < N_COMMON - 1; ++c) {
        confusion_matrix_[pred_cls][c + 1] = row[c].as<int>();
      }
    }

    // Compute per-class precision
    RCLCPP_INFO_STREAM(node_->get_logger(), "Loaded confusion matrix from " << yaml_path);
    RCLCPP_INFO_STREAM(node_->get_logger(), "Per-class precision:");
    for (int cls = 1; cls < N_COMMON; ++cls) {
      int row_total = 0;
      for (int c = 0; c < N_COMMON; ++c) row_total += confusion_matrix_[cls][c];
      float prec = (row_total > 0)
          ? static_cast<float>(confusion_matrix_[cls][cls]) / row_total
          : 0.0f;
      class_precision_[cls] = prec;
      RCLCPP_INFO_STREAM(node_->get_logger(), "  class " << cls << ": precision=" << (prec * 100.0f) << "%");
    }
    confusion_matrix_loaded_ = true;
    return true;
  } catch (const std::exception& e) {
    RCLCPP_ERROR_STREAM(node_->get_logger(), "Failed to load confusion matrix: " << e.what());
    return false;
  }
}

void CommonUtils::set_color_mode(osm_bki::MapColorMode mode) {
  if (m_pub_) m_pub_->set_color_mode(mode);
}

void CommonUtils::set_publish_variance(bool enabled, const std::string& topic) {
  publish_variance_ = enabled;
  variance_topic_ = topic;
  if (enabled && !variance_pub_) {
    variance_pub_ = new osm_bki::MarkerArrayPub(node_, topic, static_cast<float>(resolution_));
    RCLCPP_INFO_STREAM(node_->get_logger(), "Variance visualization enabled on topic: " << topic);
  }
}

void CommonUtils::set_publish_semantic_uncertainty(bool enabled, const std::string& topic) {
  publish_semantic_uncertainty_ = enabled;
  semantic_uncertainty_topic_ = topic;
  if (enabled && !semantic_uncertainty_pub_) {
    semantic_uncertainty_pub_ = new osm_bki::MarkerArrayPub(node_, topic, static_cast<float>(resolution_));
    RCLCPP_INFO_STREAM(node_->get_logger(), "Semantic uncertainty map visualization enabled on topic: " << topic);
  }
}

void CommonUtils::set_osm_buildings(const std::vector<osm_bki::Geometry2D> &buildings) {
  if (map_) map_->set_osm_buildings(buildings);
}
void CommonUtils::set_osm_roads(const std::vector<osm_bki::Geometry2D> &roads) {
  if (map_) map_->set_osm_roads(roads);
}
void CommonUtils::set_osm_sidewalks(const std::vector<osm_bki::Geometry2D> &sidewalks) {
  if (map_) map_->set_osm_sidewalks(sidewalks);
}
void CommonUtils::set_osm_cycleways(const std::vector<osm_bki::Geometry2D> &cycleways) {
  if (map_) map_->set_osm_cycleways(cycleways);
}
void CommonUtils::set_osm_grasslands(const std::vector<osm_bki::Geometry2D> &grasslands) {
  if (map_) map_->set_osm_grasslands(grasslands);
}
void CommonUtils::set_osm_trees(const std::vector<osm_bki::Geometry2D> &trees) {
  if (map_) map_->set_osm_trees(trees);
}
void CommonUtils::set_osm_forests(const std::vector<osm_bki::Geometry2D> &forests) {
  if (map_) map_->set_osm_forests(forests);
}
void CommonUtils::set_osm_tree_points(const std::vector<std::pair<float, float>> &tree_points) {
  if (map_) map_->set_osm_tree_points(tree_points);
}
void CommonUtils::set_osm_parking(const std::vector<osm_bki::Geometry2D> &parking) {
  if (map_) map_->set_osm_parking(parking);
}
void CommonUtils::set_osm_fences(const std::vector<osm_bki::Geometry2D> &fences) {
  if (map_) map_->set_osm_fences(fences);
}

void CommonUtils::set_osm_tree_point_radius(float radius_m) {
  if (map_) map_->set_osm_tree_point_radius(radius_m);
}
void CommonUtils::set_osm_road_width(float width_m) {
  if (map_) map_->set_osm_road_width(width_m);
}
void CommonUtils::set_osm_sidewalk_width(float width_m) {
  if (map_) map_->set_osm_sidewalk_width(width_m);
}
void CommonUtils::set_osm_cycleway_width(float width_m) {
  if (map_) map_->set_osm_cycleway_width(width_m);
}
void CommonUtils::set_osm_fence_width(float width_m) {
  if (map_) map_->set_osm_fence_width(width_m);
}

void CommonUtils::set_osm_decay_meters(float decay_m) {
  if (map_) map_->set_osm_decay_meters(decay_m);
}
void CommonUtils::set_osm_prior_strength(float strength) {
  if (map_) map_->set_osm_prior_strength(strength);
}
void CommonUtils::set_osm_dirichlet_prior_strength(float strength) {
  if (map_) map_->set_osm_dirichlet_prior_strength(strength);
}
void CommonUtils::set_osm_scan_radius_extension(float factor) {
  if (map_) map_->set_osm_scan_radius_extension(factor);
}

void CommonUtils::set_height_kernel_params(float lambda,
                                       const std::vector<float> &mu,
                                       const std::vector<float> &tau,
                                       const std::vector<float> &dead_zone,
                                       bool redistribute,
                                       float gate,
                                       float sensor_mounting_height) {
  if (map_) map_->set_height_kernel_params(lambda, mu, tau, dead_zone,
                                            redistribute, gate, sensor_mounting_height);
}

void CommonUtils::set_publish_osm_prior_map(bool enabled, const std::string& topic,
                                        osm_bki::MapColorMode osm_color_mode, float raster_z) {
  publish_osm_prior_map_ = enabled;
  osm_prior_map_color_mode_ = osm_color_mode;
  osm_prior_map_z_ = raster_z;
  if (enabled && !osm_prior_map_pub_) {
    osm_prior_map_pub_ = new osm_bki::MarkerArrayPub(node_, topic, static_cast<float>(resolution_));
    RCLCPP_INFO_STREAM(node_->get_logger(),
        "OSM prior map (2D raster) enabled on topic: " << topic
        << " (z=" << raster_z << ")");
  }
}

void CommonUtils::set_publish_osm_converted_map(bool enabled, const std::string& topic) {
  publish_osm_converted_map_ = enabled;
  if (enabled && !osm_converted_map_pub_) {
    osm_converted_map_pub_ = new osm_bki::MarkerArrayPub(node_, topic, static_cast<float>(resolution_));
    // Mirror the main map's semantic palette so the two are directly comparable.
    if (!colors_file_path_.empty()) {
      osm_converted_map_pub_->load_colors_from_yaml(colors_file_path_);
    }
    RCLCPP_INFO_STREAM(node_->get_logger(),
        "OSM-converted map (CM + height filter) enabled on topic: " << topic);
  }
}

bool CommonUtils::load_osm_geometry_parameters(const std::string &yaml_path) {
  if (!map_) return false;
  try {
    YAML::Node root = YAML::LoadFile(yaml_path);
    if (!root["osm_geometry_parameters"]) return false;

    auto p = root["osm_geometry_parameters"];
    bool loaded_any = false;
    auto load_float = [&](const char *key, auto setter) {
      if (p[key]) {
        setter(p[key].as<float>());
        loaded_any = true;
      }
    };

    load_float("road_width_meters", [&](float v) { map_->set_osm_road_width(v); });
    load_float("sidewalk_width_meters", [&](float v) { map_->set_osm_sidewalk_width(v); });
    load_float("cycleway_width_meters", [&](float v) { map_->set_osm_cycleway_width(v); });
    load_float("fence_width_meters", [&](float v) { map_->set_osm_fence_width(v); });
    load_float("tree_point_radius_meters", [&](float v) { map_->set_osm_tree_point_radius(v); });
    load_float("osm_decay_meters", [&](float v) { map_->set_osm_decay_meters(v); });

    if (loaded_any) {
      RCLCPP_INFO_STREAM(node_->get_logger(),
          "Loaded OSM geometry parameters from " << yaml_path);
    }
    return loaded_any;
  } catch (const std::exception &e) {
    RCLCPP_WARN_STREAM(node_->get_logger(),
        "Failed to load OSM geometry parameters: " << e.what());
    return false;
  }
}

bool CommonUtils::load_osm_confusion_matrix(const std::string &yaml_path) {
  if (!map_) return false;
  try {
    YAML::Node root = YAML::LoadFile(yaml_path);
    if (!root["confusion_matrix"] || !root["label_to_matrix_idx"])
      return false;

    auto cm_node = root["confusion_matrix"];
    auto lm_node = root["label_to_matrix_idx"];

    // Determine row count from label_to_matrix_idx
    int max_row = -1;
    for (auto it = lm_node.begin(); it != lm_node.end(); ++it)
      max_row = std::max(max_row, it->second.as<int>());
    int n_rows = max_row + 1;

    // Parse confusion matrix rows (keyed by common class ID).
    // Columns are the OSM prior categories defined in bkioctomap.h:
    //   [roads, sidewalks, cycleways, parking, grasslands, trees, forest, buildings, fences, none]
    std::vector<std::vector<float>> matrix(n_rows, std::vector<float>(osm_bki::SemanticBKIOctoMap::N_OSM_PRIOR_COLS, 0.f));
    for (auto it = cm_node.begin(); it != cm_node.end(); ++it) {
      int common_class = it->first.as<int>();
      int row = -1;
      if (lm_node[common_class]) row = lm_node[common_class].as<int>();
      if (row < 0 || row >= n_rows) continue;
      auto vals = it->second;
      for (int c = 0; c < std::min(static_cast<int>(vals.size()), osm_bki::SemanticBKIOctoMap::N_OSM_PRIOR_COLS); ++c)
        matrix[row][c] = vals[c].as<float>();
    }

    // Labels in ybars are now common taxonomy indices, so each confusion
    // matrix row maps directly to its common class ID.
    std::vector<std::vector<int>> row_to_labels(n_rows);
    for (auto it = lm_node.begin(); it != lm_node.end(); ++it) {
      int common_class = it->first.as<int>();
      int row = it->second.as<int>();
      if (common_class > 0 && row >= 0 && row < n_rows)
        row_to_labels[row].push_back(common_class);
    }

    map_->set_osm_confusion_matrix(matrix, row_to_labels);
    RCLCPP_INFO_STREAM(node_->get_logger(),
        "OSM confusion matrix loaded: " << n_rows << " rows x " << osm_bki::SemanticBKIOctoMap::N_OSM_PRIOR_COLS << " cols");
    return true;
  } catch (const std::exception &e) {
    RCLCPP_WARN_STREAM(node_->get_logger(),
        "Failed to load OSM confusion matrix: " << e.what());
    return false;
  }
}

bool CommonUtils::scan_and_label_exist(const std::string& input_data_dir, const std::string& input_label_dir, int scan_file_num) {
  char scan_id_c[256];
  std::snprintf(scan_id_c, sizeof(scan_id_c), "%010d", scan_file_num);
  std::string scan_name = input_data_dir + "/" + std::string(scan_id_c) + ".bin";
  FILE* fp = std::fopen(scan_name.c_str(), "rb");
  if (!fp) return false;
  std::fclose(fp);

  std::string label_name;
  if (inferred_use_multiclass_) {
    label_name = inferred_multiclass_dir_ + "/" + std::string(scan_id_c) + ".bin";
  } else {
    label_name = input_label_dir + "/" + std::string(scan_id_c) + ".bin";
  }
  FILE* fp_label = std::fopen(label_name.c_str(), "rb");
  if (!fp_label) return false;
  std::fclose(fp_label);
  return true;
}

bool CommonUtils::process_scans(std::string input_data_dir, std::string input_label_dir, int scan_num, double keyframe_dist, bool query, bool publish_semantic_occ_map) {
  publish_semantic_occ_map_ = publish_semantic_occ_map;
  if (!map_) {
    RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: process_scans: map_ is null!");
    return false;
  }
  if (!m_pub_) {
    RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: process_scans: m_pub_ is null!");
    return false;
  }
  if (lidar_poses_.empty()) {
    RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: process_scans: No poses loaded!");
    return false;
  }

  // Build list of pose indices where both lidar bin and label file exist (matching file names)
  std::vector<int> valid_pose_indices;
  valid_pose_indices.reserve(lidar_poses_.size());
  for (int pose_idx = 0; pose_idx < static_cast<int>(lidar_poses_.size()); ++pose_idx) {
    int scan_file_num = scan_indices_[pose_idx];
    if (scan_and_label_exist(input_data_dir, input_label_dir, scan_file_num))
      valid_pose_indices.push_back(pose_idx);
  }
  RCLCPP_INFO_STREAM(node_->get_logger(), "Found " << valid_pose_indices.size() << " scans with both lidar and label files (out of " << lidar_poses_.size() << " poses). Applying scan_num=" << scan_num << ", keyframe_dist=" << keyframe_dist << "m");

  // Apply keyframe_dist: only keep scans whose pose is at least keyframe_dist (m) from the last accepted keyframe, up to scan_num
  std::vector<int> indices_to_process;
  Eigen::Vector3d last_keyframe_pos(0, 0, 0);
  bool have_keyframe = false;
  for (size_t i = 0; i < valid_pose_indices.size(); ++i) {
    int pose_idx = valid_pose_indices[i];
    Eigen::Vector3d pos = lidar_poses_[pose_idx].block<3, 1>(0, 3);
    if (have_keyframe && keyframe_dist > 0.0) {
      double dist = (pos - last_keyframe_pos).norm();
      if (dist < keyframe_dist)
        continue;
    }
    if (static_cast<int>(indices_to_process.size()) >= scan_num)
      break;
    indices_to_process.push_back(pose_idx);
    last_keyframe_pos = pos;
    have_keyframe = true;
  }

  osm_bki::point3f origin;
  int insertion_count = 0;

  for (size_t list_idx = 0; list_idx < indices_to_process.size(); ++list_idx) {
    int pose_idx = indices_to_process[list_idx];
    int scan_file_num = scan_indices_[pose_idx];

    char scan_id_c[256];
    sprintf(scan_id_c, "%010d", scan_file_num);
    std::string scan_name = input_data_dir + "/" + std::string(scan_id_c) + ".bin";

    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud;
    pcl::PointCloud<pcl::PointXYZL>::ConstPtr orig_cloud_for_unc;
    std::vector<float> orig_variances_copy;  // copy must outlive mc_result for accumulation
    int orig_n_classes_for_unc = 0;
    if (inferred_use_multiclass_) {
      std::string mc_name = inferred_multiclass_dir_ + "/" + std::string(scan_id_c) + ".bin";
      MulticlassResult mc_result = read_scan_with_multiclass_scores(scan_name, mc_name);
      cloud = mc_result.cloud;
      scan_common_probs_ = mc_result.common_probs;

      if (publish_semantic_uncertainty_ && mc_result.n_classes > 1 && !mc_result.variances.empty()) {
        orig_cloud_for_unc = mc_result.cloud;
        orig_variances_copy = mc_result.variances;  // copy: mc_result is destroyed at block end
        orig_n_classes_for_unc = mc_result.n_classes;
      }
      // Compute per-point weights from uncertainty for kernel discounting
      if (use_uncertainty_filter_ && cloud && !cloud->points.empty() &&
          mc_result.n_classes > 1) {

        size_t n_pts = cloud->points.size();

        // Per-point uncertainties = normalized entropy (already in [0, 1]).
        const std::vector<float>& uncertainties = mc_result.variances;

        // Top-N% thresholding: keep the (1 - drop_pct)% least uncertain
        // points at full weight; ramp the rest down to uncertainty_min_weight_.
        float keep_fraction = 1.0f - uncertainty_drop_percent_ / 100.0f;
        size_t keep_rank = static_cast<size_t>(keep_fraction * n_pts);
        if (keep_rank >= n_pts) keep_rank = n_pts - 1;

        std::vector<float> sorted_unc(uncertainties);
        std::nth_element(sorted_unc.begin(),
                         sorted_unc.begin() + keep_rank,
                         sorted_unc.end());
        float threshold = sorted_unc[keep_rank];

        scan_point_weights_.resize(n_pts);
        int n_discounted = 0;
        for (size_t pi = 0; pi < n_pts; ++pi) {
          if (uncertainties[pi] <= threshold) {
            scan_point_weights_[pi] = 1.0f;
          } else {
            float denom = (1.0f - threshold);
            float t = (denom > 1e-6f)
                ? std::min((uncertainties[pi] - threshold) / denom, 1.0f)
                : 1.0f;
            scan_point_weights_[pi] = 1.0f - t * (1.0f - uncertainty_min_weight_);
            n_discounted++;
          }
        }

        total_points_processed_ += static_cast<int>(n_pts);
        total_points_filtered_ += n_discounted;
        if (n_discounted > 0 && (list_idx < 5 || list_idx % 50 == 0)) {
          RCLCPP_INFO_STREAM(node_->get_logger(),
              "Scan " << list_idx << ": discounted " << n_discounted << "/"
              << n_pts << " points (cumulative: "
              << total_points_filtered_ << "/" << total_points_processed_ << ")");
        }
      } else {
        scan_point_weights_.clear();
      }
    }
    else {
      scan_common_probs_.clear();
      std::string label_name = input_label_dir + "/" + std::string(scan_id_c) + ".bin";
      cloud = read_scan_with_labels(scan_name, label_name);
    }

    if (!cloud) {
      RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: read_scan_with_labels returned null pointer for scan " << scan_file_num);
      continue;
    }
    if (cloud->points.empty()) {
      RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Empty point cloud at scan file " << scan_file_num << " (pose index " << pose_idx << "), skipping");
      continue;
    }

    Eigen::Matrix4d transform = lidar_poses_[pose_idx];  // This is body_to_world from pose

    // Verify body-to-lidar transform is loaded (should not be zero matrix)
    if (body_to_lidar_tf_.isZero(1e-10)) {
      RCLCPP_FATAL_STREAM(node_->get_logger(), "ERROR: body_to_lidar_tf_ is not initialized! Calibration must be loaded before processing scans.");
      RCLCPP_FATAL_STREAM(node_->get_logger(), "Call load_calibration_from_params() and ensure it returns true.");
      exit(1);
    }

    // Apply body-to-lidar transformation
    // The poses in CSV are body/IMU poses, need to transform from lidar frame to world frame
    // Following Python code: transform_matrix = body_to_world @ lidar_to_body
    // where lidar_to_body = inv(body_to_lidar_tf)
    Eigen::Matrix4d lidar_to_body = body_to_lidar_tf_.inverse();
    Eigen::Matrix4d lidar_to_map = transform * lidar_to_body;  // T_lidar_to_map = body_to_world * lidar_to_body

    // Publish TF transform from 'map' to 'lidar' frame
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = node_->now();
    t.header.frame_id = "map";
    t.child_frame_id = "lidar";

    Eigen::Matrix3d rotation = lidar_to_map.block<3, 3>(0, 0);
    Eigen::Vector3d translation = lidar_to_map.block<3, 1>(0, 3);

    Eigen::Quaterniond quat(rotation);
    t.transform.translation.x = translation(0);
    t.transform.translation.y = translation(1);
    t.transform.translation.z = translation(2);
    t.transform.rotation.x = quat.x();
    t.transform.rotation.y = quat.y();
    t.transform.rotation.z = quat.z();
    t.transform.rotation.w = quat.w();

    tf_broadcaster_.sendTransform(t);

    // Debug: Print transform info for first few processed scans
    if (list_idx < 3) {
      RCLCPP_INFO_STREAM(node_->get_logger(), "Scan " << list_idx << " (pose_idx " << pose_idx << ") Transform info:");
      RCLCPP_INFO_STREAM(node_->get_logger(), "  Body-to-world translation from CSV: [" << transform(0,3) << ", " << transform(1,3) << ", " << transform(2,3) << "]");
      RCLCPP_INFO_STREAM(node_->get_logger(), "  Lidar-to-map translation: [" << lidar_to_map(0,3) << ", " << lidar_to_map(1,3) << ", " << lidar_to_map(2,3) << "]");
      RCLCPP_INFO_STREAM(node_->get_logger(), "  TF translation (from lidar_to_map): [" << translation(0) << ", " << translation(1) << ", " << translation(2) << "]");

      // Verify: transform origin from lidar frame should give lidar_to_map translation
      Eigen::Vector4d lidar_origin(0, 0, 0, 1);
      Eigen::Vector4d map_origin_test = lidar_to_map * lidar_origin;
      RCLCPP_INFO_STREAM(node_->get_logger(), "  Verification - lidar origin in map coords: [" << map_origin_test(0) << ", " << map_origin_test(1) << ", " << map_origin_test(2) << "]");
    }

    // Transform cloud from lidar frame to map frame (once, before publish and insertion)
    pcl::transformPointCloud(*cloud, *cloud, lidar_to_map.cast<float>());

    // Publish scan as PointCloud2 in map frame with label-based RGB colors
    {
      pcl::PointCloud<pcl::PointXYZRGB> rgb_cloud;
      rgb_cloud.width = static_cast<uint32_t>(cloud->points.size());
      rgb_cloud.height = 1;
      rgb_cloud.is_dense = true;
      rgb_cloud.points.resize(cloud->points.size());
      for (size_t i = 0; i < cloud->points.size(); ++i) {
        rgb_cloud.points[i].x = cloud->points[i].x;
        rgb_cloud.points[i].y = cloud->points[i].y;
        rgb_cloud.points[i].z = cloud->points[i].z;
        std_msgs::msg::ColorRGBA color = m_pub_->get_color_for_class(
            static_cast<int>(cloud->points[i].label), 2);
        rgb_cloud.points[i].r = static_cast<uint8_t>(std::min(255, static_cast<int>(color.r * 255.0f + 0.5f)));
        rgb_cloud.points[i].g = static_cast<uint8_t>(std::min(255, static_cast<int>(color.g * 255.0f + 0.5f)));
        rgb_cloud.points[i].b = static_cast<uint8_t>(std::min(255, static_cast<int>(color.b * 255.0f + 0.5f)));
      }
      sensor_msgs::msg::PointCloud2 cloud_msg;
      pcl::toROSMsg(rgb_cloud, cloud_msg);
      cloud_msg.header.frame_id = "map";
      cloud_msg.header.stamp = node_->now();
      pointcloud_pub_->publish(cloud_msg);
    }

    // Accumulate semantic uncertainty per voxel (for map visualization)
    if (orig_cloud_for_unc && !orig_variances_copy.empty() && publish_semantic_uncertainty_) {
      float max_var = static_cast<float>(orig_n_classes_for_unc - 1) /
          (static_cast<float>(orig_n_classes_for_unc) * orig_n_classes_for_unc);
      pcl::PointCloud<pcl::PointXYZL> cloud_map;
      pcl::transformPointCloud(*orig_cloud_for_unc, cloud_map, lidar_to_map.cast<float>());
      for (size_t pi = 0; pi < cloud_map.points.size(); ++pi) {
        float u = 1.0f - std::min(orig_variances_copy[pi] / max_var, 1.0f);
        int kx = static_cast<int>(std::floor(cloud_map.points[pi].x / resolution_));
        int ky = static_cast<int>(std::floor(cloud_map.points[pi].y / resolution_));
        int kz = static_cast<int>(std::floor(cloud_map.points[pi].z / resolution_));
        VoxelKey key{kx, ky, kz};
        auto& p = semantic_uncertainty_acc_[key];
        p.first += u;
        p.second += 1;
      }
    }

    if (insertion_count == 0) {
      RCLCPP_INFO_STREAM(node_->get_logger(), "Published PointCloud2 with " << cloud->points.size() << " points in frame 'map'");
    }

    // Cloud is already in map frame. Sensor origin from lidar_to_map translation.
    origin.x() = lidar_to_map(0, 3);
    origin.y() = lidar_to_map(1, 3);
    origin.z() = lidar_to_map(2, 3);

    try {
      if (!scan_common_probs_.empty() && scan_common_probs_.size() == cloud->points.size()) {
        map_->insert_pointcloud(*cloud, origin, ds_resolution_, free_resolution_, max_range_,
                                scan_point_weights_, &scan_common_probs_);
      } else if (!scan_point_weights_.empty() && scan_point_weights_.size() == cloud->points.size()) {
        map_->insert_pointcloud(*cloud, origin, ds_resolution_, free_resolution_, max_range_, scan_point_weights_);
      } else {
        map_->insert_pointcloud(*cloud, origin, ds_resolution_, free_resolution_, max_range_);
      }
    } catch (const std::exception& e) {
      RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Exception during insert_pointcloud: " << e.what());
      continue;
    }
    insertion_count++;

    // Capture this scan's xy footprint (cloud is in the map frame). Used by the
    // OSM prior-map raster so its extent matches the current scan, not the cumulative map.
    if (!cloud->points.empty()) {
      float min_x = std::numeric_limits<float>::infinity();
      float min_y = std::numeric_limits<float>::infinity();
      float max_x = -std::numeric_limits<float>::infinity();
      float max_y = -std::numeric_limits<float>::infinity();
      for (const auto& pt : cloud->points) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y)) continue;
        if (pt.x < min_x) min_x = pt.x;
        if (pt.y < min_y) min_y = pt.y;
        if (pt.x > max_x) max_x = pt.x;
        if (pt.y > max_y) max_y = pt.y;
      }
      if (std::isfinite(min_x) && std::isfinite(max_x) && max_x > min_x && max_y > min_y) {
        scan_xy_min_x_ = min_x;
        scan_xy_min_y_ = min_y;
        scan_xy_max_x_ = max_x;
        scan_xy_max_y_ = max_y;
        scan_xy_set_ = true;
      }
    }

    // Skip query/visualize for first insertion to avoid potential segfaults with empty/initializing octree
    if (insertion_count == 1) {
      RCLCPP_DEBUG_STREAM(node_->get_logger(), "Skipping query/visualize for first insertion");
      continue;
    }

    if (query) {
      // Query previous scans (use pose indices, not file numbers)
      // Original ROS1 logic: for (int query_pose_idx = pose_idx - 10; query_pose_idx >= 0 && query_pose_idx <= pose_idx; ++query_pose_idx)
      for (int query_pose_idx = pose_idx - 10; query_pose_idx >= 0 && query_pose_idx <= pose_idx; ++query_pose_idx) {
        query_scan(input_data_dir, input_label_dir, query_pose_idx);
      }
    }

    if (any_publish_enabled()) {
      publish_map();
      // Small delay to allow rviz to process the visualization
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  // Final publish after all scans are processed
  if (any_publish_enabled()) {
    RCLCPP_INFO_STREAM(node_->get_logger(), "All scans processed. Publishing final map visualization...");
    publish_map();
  }

  return true;
}

bool CommonUtils::any_publish_enabled() const {
  return publish_semantic_occ_map_
      || publish_variance_
      || publish_osm_prior_map_
      || publish_osm_converted_map_
      || publish_semantic_uncertainty_;
}

void CommonUtils::publish_map() {
  if (!m_pub_) {
    RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: publish_map: m_pub_ is null!");
    return;
  }
  if (!map_) {
    RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: publish_map: map_ is null!");
    return;
  }

  if (publish_semantic_occ_map_) {
    m_pub_->clear_map(resolution_);
  }

  // Check if map is empty before iterating - get iterators separately to catch segfault location
  try {
    auto begin_it = map_->begin_leaf();

    auto end_it = map_->end_leaf();

    if (begin_it == end_it) {
      RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Map is empty (begin == end), nothing to publish");
      if (publish_semantic_occ_map_) m_pub_->publish();
      return;
    }
  } catch (const std::exception& e) {
    RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Exception getting iterators: " << e.what());
    return;
  } catch (...) {
    RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Unknown exception getting iterators");
    return;
  }

  // First pass: main semantic map (always semantic colors) + compute min/max variance for variance visualization
  float min_var = std::numeric_limits<float>::max();
  float max_var = std::numeric_limits<float>::lowest();
  int voxel_count = 0;
  int iter_count = 0;
  try {
    auto loop_begin = map_->begin_leaf();
    auto loop_end = map_->end_leaf();

    auto it = loop_begin;

    while (it != loop_end) {
      iter_count++;

      try {
        auto node = it.get_node();

          if (node.get_state() == osm_bki::State::OCCUPIED) {
          osm_bki::point3f p = it.get_loc();
          float size = it.get_size();
          if (publish_variance_) {
            std::vector<float> vars(num_class_);
            node.get_vars(vars);
            int semantics = node.get_semantics();
            if (semantics >= 0 && semantics < num_class_) {
              float v = vars[semantics];
              if (v > max_var) max_var = v;
              if (v < min_var) min_var = v;
            }
          }
          if (publish_semantic_occ_map_) {
            m_pub_->insert_point3d_semantics(p.x(), p.y(), p.z(), size, node.get_semantics(), 2);
          }
          voxel_count++;
        }

        if (iter_count % 1000 == 0) {
          RCLCPP_DEBUG_STREAM(node_->get_logger(), "Processed " << iter_count << " nodes");
        }

        // Increment iterator
        ++it;
      } catch (const std::exception& e) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Exception at iteration " << iter_count << ": " << e.what());
        break;
      } catch (...) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Unknown exception at iteration " << iter_count);
        break;
      }
    }
  } catch (const std::exception& e) {
    RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Exception during map iteration: " << e.what());
    return;
  }
  (void)voxel_count;
  if (publish_semantic_occ_map_) {
    m_pub_->publish();
  }

  // Second pass: publish variance map if enabled
  if (publish_variance_ && variance_pub_) {
    if (min_var > max_var) min_var = max_var = 0.f;  // avoid div by zero
    variance_pub_->clear_map(static_cast<float>(resolution_));
    for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it) {
      auto node = it.get_node();
      if (node.get_state() == osm_bki::State::OCCUPIED) {
        osm_bki::point3f p = it.get_loc();
        int semantics = node.get_semantics();
        std::vector<float> vars(num_class_);
        node.get_vars(vars);
        if (semantics >= 0 && semantics < num_class_)
          variance_pub_->insert_point3d_variance(p.x(), p.y(), p.z(), min_var, max_var, it.get_size(), vars[semantics]);
      }
    }
    variance_pub_->publish();
  }

  // OSM prior map (2D raster): sample a regular xy grid over the *current scan's*
  // xy footprint at a single z layer; raw OSM priors, no height filter. The
  // footprint is captured during insertion so the raster doesn't grow with the map.
  if (publish_osm_prior_map_ && osm_prior_map_pub_ && map_ && scan_xy_set_) {
    osm_prior_map_pub_->clear_map(static_cast<float>(resolution_));
    osm_prior_map_pub_->set_color_mode(osm_prior_map_color_mode_);
    osm_bki::MapColorMode mode = osm_prior_map_color_mode_;

    const float min_x = scan_xy_min_x_;
    const float min_y = scan_xy_min_y_;
    const float max_x = scan_xy_max_x_;
    const float max_y = scan_xy_max_y_;
    const float step = static_cast<float>(resolution_);
    const float z_layer = osm_prior_map_z_;
    const float size = step;
    if (max_x > min_x && max_y > min_y && step > 0.f) {
      for (float x = min_x; x <= max_x; x += step) {
        for (float y = min_y; y <= max_y; y += step) {
          float building, road, grassland, tree, parking, fence, sidewalk, cycleway, forest;
          map_->get_osm_priors_for_visualization(x, y, building, road, grassland,
                                                  tree, parking, fence,
                                                  sidewalk, cycleway, forest);
          if (mode == osm_bki::MapColorMode::OSMBlend) {
            osm_prior_map_pub_->insert_point3d_osm_blend(x, y, z_layer, size,
                building, road, grassland, tree, parking, fence,
                sidewalk, cycleway, forest);
          } else {
            int prior_type = 0;
            float value = 0.f;
            switch (mode) {
              case osm_bki::MapColorMode::OSMBuilding:   prior_type = 0; value = building; break;
              case osm_bki::MapColorMode::OSMRoad:       prior_type = 1; value = road; break;
              case osm_bki::MapColorMode::OSMGrassland:  prior_type = 2; value = grassland; break;
              case osm_bki::MapColorMode::OSMTree:       prior_type = 3; value = tree; break;
              case osm_bki::MapColorMode::OSMParking:    prior_type = 4; value = parking; break;
              case osm_bki::MapColorMode::OSMFence:      prior_type = 5; value = fence; break;
              case osm_bki::MapColorMode::OSMSidewalk:   prior_type = 6; value = sidewalk; break;
              case osm_bki::MapColorMode::OSMCycleway:   prior_type = 7; value = cycleway; break;
              case osm_bki::MapColorMode::OSMForest:     prior_type = 8; value = forest; break;
              default: prior_type = 0; value = building; break;
            }
            osm_prior_map_pub_->insert_point3d_osm_prior(x, y, z_layer, size, value, prior_type);
          }
        }
      }
    }
    osm_prior_map_pub_->publish();
  }

  // OSM-converted map: per-occupied-voxel argmax of the OSM CM projection with the
  // active height filter applied. Colored with the semantic palette so the result
  // is directly comparable to the main semantic occupancy map.
  if (publish_osm_converted_map_ && osm_converted_map_pub_ && map_) {
    osm_converted_map_pub_->clear_map(static_cast<float>(resolution_));
    std::vector<float> common_priors;
    for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it) {
      auto node = it.get_node();
      if (node.get_state() != osm_bki::State::OCCUPIED) continue;
      osm_bki::point3f p = it.get_loc();
      float size = it.get_size();
      if (!map_->compute_osm_converted_prior(p.x(), p.y(), p.z(), common_priors)) continue;

      int best = 0;
      float best_val = common_priors.empty() ? 0.f : common_priors[0];
      for (int k = 1; k < static_cast<int>(common_priors.size()); ++k) {
        if (common_priors[k] > best_val) {
          best_val = common_priors[k];
          best = k;
        }
      }
      // Skip when the argmax falls on the unlabeled / zero-prior bucket.
      if (best <= 0 || best_val <= 0.f) continue;
      osm_converted_map_pub_->insert_point3d_semantics(p.x(), p.y(), p.z(), size, best, 2);
    }
    osm_converted_map_pub_->publish();
  }

  // Fourth pass: publish semantic uncertainty map if enabled (input observation uncertainty per voxel)
  if (publish_semantic_uncertainty_ && semantic_uncertainty_pub_ && !semantic_uncertainty_acc_.empty()) {
    float min_unc = 1.f;
    float max_unc = 0.f;
    for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it) {
      auto node = it.get_node();
      if (node.get_state() != osm_bki::State::OCCUPIED) continue;
      osm_bki::point3f p = it.get_loc();
      int kx = static_cast<int>(std::floor(p.x() / resolution_));
      int ky = static_cast<int>(std::floor(p.y() / resolution_));
      int kz = static_cast<int>(std::floor(p.z() / resolution_));
      VoxelKey key{kx, ky, kz};
      auto fit = semantic_uncertainty_acc_.find(key);
      if (fit == semantic_uncertainty_acc_.end()) continue;
      float avg_u = fit->second.first / static_cast<float>(fit->second.second);
      if (avg_u < min_unc) min_unc = avg_u;
      if (avg_u > max_unc) max_unc = avg_u;
    }
    if (min_unc > max_unc) min_unc = max_unc = 0.f;
    semantic_uncertainty_pub_->clear_map(static_cast<float>(resolution_));
    for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it) {
      auto node = it.get_node();
      if (node.get_state() != osm_bki::State::OCCUPIED) continue;
      osm_bki::point3f p = it.get_loc();
      int kx = static_cast<int>(std::floor(p.x() / resolution_));
      int ky = static_cast<int>(std::floor(p.y() / resolution_));
      int kz = static_cast<int>(std::floor(p.z() / resolution_));
      VoxelKey key{kx, ky, kz};
      auto fit = semantic_uncertainty_acc_.find(key);
      if (fit == semantic_uncertainty_acc_.end()) continue;
      float avg_u = fit->second.first / static_cast<float>(fit->second.second);
      semantic_uncertainty_pub_->insert_point3d_variance(p.x(), p.y(), p.z(), min_unc, max_unc, it.get_size(), avg_u);
    }
    semantic_uncertainty_pub_->publish();
  }
}

bool CommonUtils::load_colors_from_params() {
  if (m_pub_) {
    return m_pub_->load_colors_from_params(node_);
  }
  return false;
}

bool CommonUtils::load_colors_from_yaml(const std::string& yaml_file_path) {
  RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: CommonUtils::load_colors_from_yaml: Starting, file=" << yaml_file_path);
  if (!m_pub_) {
    RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: m_pub_ is null, cannot load colors!");
    return false;
  }
  RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: m_pub_ is valid, calling load_colors_from_yaml");
  bool result = m_pub_->load_colors_from_yaml(yaml_file_path);
  if (result) colors_file_path_ = yaml_file_path;  // cache for downstream pubs that share semantic colors
  // Apply to any already-created semantic-colored publishers (e.g. osm_converted_map_pub_).
  if (result && osm_converted_map_pub_) {
    osm_converted_map_pub_->load_colors_from_yaml(yaml_file_path);
  }
  RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: CommonUtils::load_colors_from_yaml: Result=" << result);
  return result;
}

void CommonUtils::set_up_evaluation(const std::string gt_label_dir, const std::string evaluation_result_dir) {
  gt_label_dir_ = gt_label_dir;
  evaluation_result_dir_ = evaluation_result_dir;
}

bool CommonUtils::load_calibration_from_params() {
  // Load body-to-lidar transform from ROS parameters
  // Expected path: body/os_sensor/T (from hhs_calib.yaml)
  // If calibration_file is empty (e.g. KITTI360), use identity (no body-to-lidar transform).
  RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: load_calibration_from_params: Starting");
  try {
    std::string calib_file;
    if (node_->get_parameter("calibration_file", calib_file)) {
      if (calib_file.empty()) {
        body_to_lidar_tf_ = Eigen::Matrix4d::Identity();
        RCLCPP_INFO_STREAM(node_->get_logger(), "No calibration file (empty); using identity body-to-lidar transform.");
        return true;
      }
      RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Found calibration_file parameter: " << calib_file);
      RCLCPP_INFO_STREAM(node_->get_logger(), "Loading calibration from YAML file: " << calib_file);
      bool result = load_calibration_from_yaml(calib_file);
      RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: load_calibration_from_yaml returned: " << result);
      return result;
    }
    RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: calibration_file parameter not found");

    if (node_->has_parameter("body")) {
      RCLCPP_WARN_STREAM(node_->get_logger(), "Calibration loading from ROS parameters not fully implemented. Please use calibration_file parameter.");
      return false;
    }

    RCLCPP_ERROR_STREAM(node_->get_logger(), "ERROR: 'calibration_file' parameter not found in ROS parameter server!");
    return false;
  } catch (const std::exception& e) {
    RCLCPP_ERROR_STREAM(node_->get_logger(), "Error loading calibration: " << e.what());
    return false;
  }
}

bool CommonUtils::load_calibration_from_yaml(const std::string& yaml_file) {
  RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: load_calibration_from_yaml: Loading file: " << yaml_file);
  try {
    YAML::Node yaml_node = YAML::LoadFile(yaml_file);
    RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: YAML file loaded successfully");
    if (!yaml_node["body"]) {
      RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: 'body' key not found in YAML file!");
      RCLCPP_ERROR_STREAM(node_->get_logger(), "ERROR: 'body' key not found in YAML file!");
      return false;
    }
    RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Found 'body' key in YAML");

    YAML::Node body_node = yaml_node["body"];
    if (!body_node["os_sensor"]) {
      RCLCPP_ERROR_STREAM(node_->get_logger(), "ERROR: 'body/os_sensor' not found in calibration!");
      return false;
    }

    YAML::Node os_sensor_node = body_node["os_sensor"];
    if (!os_sensor_node["T"]) {
      RCLCPP_ERROR_STREAM(node_->get_logger(), "ERROR: 'body/os_sensor/T' not found in calibration!");
      return false;
    }

    YAML::Node T_node = os_sensor_node["T"];
    if (!T_node.IsSequence() || T_node.size() != 4) {
      RCLCPP_ERROR_STREAM(node_->get_logger(), "ERROR: 'body/os_sensor/T' must be an array of 4 arrays!");
      return false;
    }

    // Parse the 4x4 matrix
    for (int i = 0; i < 4; ++i) {
      if (!T_node[i].IsSequence() || T_node[i].size() != 4) {
        RCLCPP_ERROR_STREAM(node_->get_logger(), "ERROR: Row " << i << " of body/os_sensor/T must be an array of 4 elements!");
        return false;
      }
      for (int j = 0; j < 4; ++j) {
        body_to_lidar_tf_(i, j) = T_node[i][j].as<double>();
      }
    }

    RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Successfully parsed calibration transform matrix");
    RCLCPP_INFO_STREAM(node_->get_logger(), "Successfully loaded body-to-lidar transform from body/os_sensor/T");
    RCLCPP_INFO_STREAM(node_->get_logger(), "Transform matrix:");
    RCLCPP_INFO_STREAM(node_->get_logger(), "  [" << body_to_lidar_tf_(0, 0) << ", " << body_to_lidar_tf_(0, 1) << ", " << body_to_lidar_tf_(0, 2) << ", " << body_to_lidar_tf_(0, 3) << "]");
    RCLCPP_INFO_STREAM(node_->get_logger(), "  [" << body_to_lidar_tf_(1, 0) << ", " << body_to_lidar_tf_(1, 1) << ", " << body_to_lidar_tf_(1, 2) << ", " << body_to_lidar_tf_(1, 3) << "]");
    RCLCPP_INFO_STREAM(node_->get_logger(), "  [" << body_to_lidar_tf_(2, 0) << ", " << body_to_lidar_tf_(2, 1) << ", " << body_to_lidar_tf_(2, 2) << ", " << body_to_lidar_tf_(2, 3) << "]");
    RCLCPP_INFO_STREAM(node_->get_logger(), "  [" << body_to_lidar_tf_(3, 0) << ", " << body_to_lidar_tf_(3, 1) << ", " << body_to_lidar_tf_(3, 2) << ", " << body_to_lidar_tf_(3, 3) << "]");

    RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: load_calibration_from_yaml completed successfully");
    return true;
  } catch (const std::exception& e) {
    RCLCPP_ERROR_STREAM(node_->get_logger(), "Error loading calibration from YAML: " << e.what());
    return false;
  }
}

void CommonUtils::query_scan(std::string input_data_dir, std::string /* input_label_dir */, int pose_idx) {
  if (pose_idx < 0 || pose_idx >= (int)lidar_poses_.size()) {
    return;
  }

  if (!map_) {
    RCLCPP_WARN_STREAM(node_->get_logger(), "Cannot query scan: map_ is null");
    return;
  }

  try {
    // Get the actual scan file number from CSV
    int scan_file_num = scan_indices_[pose_idx];

    // Use 10-digit format for MCD file naming
    char scan_id_c[256];
    sprintf(scan_id_c, "%010d", scan_file_num);
    // Helper function to join paths (handles trailing slashes)
    auto join_path = [](const std::string& dir, const std::string& file) -> std::string {
      if (dir.empty()) return file;
      if (dir.back() == '/') return dir + file;
      return dir + "/" + file;
    };
    std::string scan_name = join_path(input_data_dir, std::string(scan_id_c) + ".bin");
    std::string gt_name = join_path(gt_label_dir_, std::string(scan_id_c) + ".bin");
    std::string result_name = join_path(evaluation_result_dir_, std::string(scan_id_c) + ".txt");

    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud;
    if (gt_use_multiclass_)
      cloud = read_gt_scan_multiclass(scan_name, gt_name);
    else
      cloud = read_scan_with_labels(scan_name, gt_name, /*use_gt_mapping=*/true);
    if (cloud->points.empty()) {
      return;
    }

    Eigen::Matrix4d transform = lidar_poses_[pose_idx];  // This is body_to_world from pose

    // Apply body-to-lidar transformation (same as in process_scans)
    // Following Python code: transform_matrix = body_to_world @ lidar_to_body
    Eigen::Matrix4d lidar_to_body = body_to_lidar_tf_.inverse();
    Eigen::Matrix4d new_transform = transform * lidar_to_body;  // body_to_world * lidar_to_body
    pcl::transformPointCloud(*cloud, *cloud, new_transform);

    // Create directory if it doesn't exist
    size_t last_slash = result_name.find_last_of('/');
    if (last_slash != std::string::npos) {
      std::string dir_path = result_name.substr(0, last_slash);
      // Try to create directory (mkdir -p equivalent)
      std::string mkdir_cmd = "mkdir -p " + dir_path;
      int result = std::system(mkdir_cmd.c_str());
      if (result != 0) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "Failed to create directory: " << dir_path);
      }
    }

    std::ofstream result_file;
    result_file.open(result_name);
    if (!result_file.is_open()) {
      RCLCPP_WARN_STREAM(node_->get_logger(), "Cannot open result file: " << result_name);
      return;
    }

    for (int i = 0; i < (int)cloud->points.size(); ++i) {
      try {
        osm_bki::SemanticOcTreeNode node = map_->search(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
        int pred_label = 0;
        if (node.get_state() == osm_bki::State::OCCUPIED)
          pred_label = node.get_semantics();
        result_file << (int)cloud->points[i].label << " " << pred_label << "\n";
      } catch (const std::exception& e) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "Exception searching map for point " << i << ": " << e.what());
        continue;
      }
    }
    result_file.close();
  } catch (const std::exception& e) {
    RCLCPP_ERROR_STREAM(node_->get_logger(), "Exception in query_scan(): " << e.what());
  } catch (...) {
    RCLCPP_ERROR_STREAM(node_->get_logger(), "Unknown exception in query_scan()");
  }
}

pcl::PointCloud<pcl::PointXYZL>::Ptr CommonUtils::read_scan_with_labels(std::string fn, std::string fn_label, bool use_gt_mapping) {
  // Open scan file
  FILE* fp = std::fopen(fn.c_str(), "rb");
  if (!fp) {
    RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Cannot open scan file: " << fn);
    return pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>);
  }

  // Open label file
  FILE* fp_label = std::fopen(fn_label.c_str(), "rb");
  if (!fp_label) {
    RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Cannot open label file: " << fn_label);
    std::fclose(fp);
    return pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>);
  }

  // Get file size for scan (x, y, z, intensity = 4 floats per point)
  std::fseek(fp, 0L, SEEK_END);
  size_t sz = std::ftell(fp);
  std::rewind(fp);
  int n_hits = static_cast<int>(sz / (sizeof(float) * 4));

  // Get label file size (expected: n_hits * sizeof(uint32_t) per label)
  std::fseek(fp_label, 0L, SEEK_END);
  long label_file_sz = std::ftell(fp_label);
  std::rewind(fp_label);
  int num_labels_in_file = static_cast<int>(label_file_sz / sizeof(uint32_t));
  (void)num_labels_in_file;

  // Preallocate point cloud for better performance (avoids reallocation)
  pcl::PointCloud<pcl::PointXYZL>::Ptr pc(new pcl::PointCloud<pcl::PointXYZL>);
  if (!pc) {
    RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Failed to allocate point cloud!");
    std::fclose(fp);
    std::fclose(fp_label);
    return pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>);
  }
  pc->points.reserve(n_hits);  // Preallocate to avoid reallocation overhead
  pc->width = n_hits;
  pc->height = 1;
  pc->is_dense = false;

  // Read data in a tighter loop; collect unique class IDs for logging
  int points_read = 0;
  std::set<int> unique_labels;
  for (int i = 0; i < n_hits; i++) {
    pcl::PointXYZL point;
    float intensity;
    uint32_t label;

    // Read point data (x, y, z, intensity as floats)
    if (fread(&point.x, sizeof(float), 1, fp) != 1) break;
    if (fread(&point.y, sizeof(float), 1, fp) != 1) break;
    if (fread(&point.z, sizeof(float), 1, fp) != 1) break;
    if (fread(&intensity, sizeof(float), 1, fp) != 1) break;

    // Read label (uint32) and map to common taxonomy
    if (fread(&label, sizeof(uint32_t), 1, fp_label) != 1) break;

    int raw = static_cast<int>(label);
    if (common_label_config_loaded_) {
      const auto& mapping = use_gt_mapping ? gt_to_common_ : inferred_to_common_;
      auto it = mapping.find(raw);
      point.label = (it != mapping.end()) ? it->second : 0;
    } else {
      point.label = raw;
    }
    unique_labels.insert(point.label);
    pc->points.push_back(point);
    points_read++;
  }

  std::fclose(fp);
  std::fclose(fp_label);

  return pc;
}

MulticlassResult CommonUtils::read_scan_with_multiclass_scores(
    const std::string& fn, const std::string& fn_multiclass) {

  MulticlassResult result;
  result.cloud.reset(new pcl::PointCloud<pcl::PointXYZL>);

  FILE* fp = std::fopen(fn.c_str(), "rb");
  if (!fp) {
    RCLCPP_WARN_STREAM(node_->get_logger(),
        "Cannot open scan file: " << fn);
    return result;
  }

  FILE* fp_mc = std::fopen(fn_multiclass.c_str(), "rb");
  if (!fp_mc) {
    RCLCPP_WARN_STREAM(node_->get_logger(),
        "Cannot open multiclass file: " << fn_multiclass);
    std::fclose(fp);
    return result;
  }

  std::fseek(fp, 0L, SEEK_END);
  size_t scan_sz = std::ftell(fp);
  std::rewind(fp);
  int n_points = static_cast<int>(scan_sz / (sizeof(float) * 4));

  std::fseek(fp_mc, 0L, SEEK_END);
  size_t mc_sz = std::ftell(fp_mc);
  std::rewind(fp_mc);
  int n_mc_values = static_cast<int>(mc_sz / sizeof(uint16_t));

  if (n_points == 0 || n_mc_values == 0) {
    std::fclose(fp);
    std::fclose(fp_mc);
    return result;
  }

  int n_classes = n_mc_values / n_points;
  if (n_mc_values != n_points * n_classes) {
    RCLCPP_ERROR_STREAM(node_->get_logger(),
        "Multiclass file size mismatch: " << n_mc_values
        << " values for " << n_points << " points (not divisible)");
    std::fclose(fp);
    std::fclose(fp_mc);
    return result;
  }
  result.n_classes = n_classes;

  RCLCPP_INFO_STREAM(node_->get_logger(),
      "Multiclass: " << n_points << " points x " << n_classes << " classes");

  std::vector<uint16_t> mc_raw(n_mc_values);
  if (std::fread(mc_raw.data(), sizeof(uint16_t), n_mc_values, fp_mc)
      != static_cast<size_t>(n_mc_values)) {
    RCLCPP_ERROR_STREAM(node_->get_logger(),
        "Failed to read multiclass data from " << fn_multiclass);
    std::fclose(fp);
    std::fclose(fp_mc);
    return result;
  }
  std::fclose(fp_mc);

  auto& pc = result.cloud;
  pc->points.reserve(n_points);
  pc->width = n_points;
  pc->height = 1;
  pc->is_dense = false;
  result.variances.reserve(n_points);
  result.common_probs.reserve(static_cast<size_t>(n_points));

  std::vector<float> raw_vals(static_cast<size_t>(n_classes));
  for (int i = 0; i < n_points; i++) {
    pcl::PointXYZL point;
    float intensity;
    if (std::fread(&point.x, sizeof(float), 1, fp) != 1) break;
    if (std::fread(&point.y, sizeof(float), 1, fp) != 1) break;
    if (std::fread(&point.z, sizeof(float), 1, fp) != 1) break;
    if (std::fread(&intensity, sizeof(float), 1, fp) != 1) break;

    const uint16_t* row_ptr = mc_raw.data() + static_cast<size_t>(i) * n_classes;

    // Convert float16 → float32, find argmax, and compute variance
    int best_class = 0;
    float best_val = half_to_float(row_ptr[0]);
    raw_vals[0] = best_val;
    float sum = best_val;
    float sum_sq = best_val * best_val;
    for (int c = 1; c < n_classes; c++) {
      float v = half_to_float(row_ptr[c]);
      raw_vals[static_cast<size_t>(c)] = v;
      sum += v;
      sum_sq += v * v;
      if (v > best_val) {
        best_val = v;
        best_class = c;
      }
    }
    float mean = sum / n_classes;
    float variance = sum_sq / n_classes - mean * mean;
    if (variance < 0.0f) variance = 0.0f;
    (void)variance;

    // Softmax over network classes (used for both common and native paths)
    float max_val = raw_vals[0];
    for (int c = 1; c < n_classes; c++) {
      if (raw_vals[static_cast<size_t>(c)] > max_val)
        max_val = raw_vals[static_cast<size_t>(c)];
    }
    float softmax_sum = 0.f;
    std::vector<float> softmax_net(static_cast<size_t>(n_classes));
    for (int c = 0; c < n_classes; c++) {
      float v = std::exp(raw_vals[static_cast<size_t>(c)] - max_val);
      softmax_net[static_cast<size_t>(c)] = v;
      softmax_sum += v;
    }
    if (softmax_sum > 1e-10f) {
      for (int c = 0; c < n_classes; c++)
        softmax_net[static_cast<size_t>(c)] /= softmax_sum;
    }


    // Softmax-normalize raw_vals, then compute normalized entropy in [0, 1].
    float max_logit = raw_vals[0];
    for (int c = 1; c < n_classes; ++c)
        if (raw_vals[c] > max_logit) max_logit = raw_vals[c];

    float Z = 0.0f;
    std::vector<float> probs(n_classes);
    for (int c = 0; c < n_classes; ++c) {
        probs[c] = std::exp(raw_vals[c] - max_logit);
        Z += probs[c];
    }
    float H = 0.0f;
    for (int c = 0; c < n_classes; ++c) {
        float p = probs[c] / Z;
        if (p > 1e-12f) H -= p * std::log(p);
    }
    float H_norm = H / std::log(static_cast<float>(n_classes));
    result.variances.push_back(H_norm);


    if (common_label_config_loaded_) {
      // Map to common taxonomy: point label and soft counts in common space
      int raw_label = best_class;
      if (!learning_map_inv_.empty()) {
        auto it_inv = learning_map_inv_.find(best_class);
        if (it_inv != learning_map_inv_.end()) raw_label = it_inv->second;
      }
      int common_label = raw_label;
      auto it_c = inferred_to_common_.find(raw_label);
      if (it_c != inferred_to_common_.end()) common_label = it_c->second;
      else common_label = 0;
      point.label = static_cast<uint32_t>(common_label);
      std::vector<float> common_probs_row(static_cast<size_t>(N_COMMON), 0.f);
      std::vector<int> common_count(static_cast<size_t>(N_COMMON), 0);
      for (int k = 0; k < n_classes; k++) {
        int r = k;
        if (!learning_map_inv_.empty()) {
          auto it_inv = learning_map_inv_.find(k);
          if (it_inv != learning_map_inv_.end()) r = it_inv->second;
        }
        it_c = inferred_to_common_.find(r);
        if (it_c == inferred_to_common_.end()) continue;
        int c_common = it_c->second;
        if (c_common >= 0 && c_common < N_COMMON) {
          common_probs_row[static_cast<size_t>(c_common)] += softmax_net[static_cast<size_t>(k)];
          common_count[static_cast<size_t>(c_common)] += 1;
        }
      }
      for (int c = 0; c < N_COMMON; ++c) {
        if (common_count[static_cast<size_t>(c)] > 0)
          common_probs_row[static_cast<size_t>(c)] /= static_cast<float>(common_count[static_cast<size_t>(c)]);
      }
      float row_sum = 0.f;
      for (int c = 0; c < N_COMMON; ++c) row_sum += common_probs_row[static_cast<size_t>(c)];
      if (row_sum > 1e-10f)
        for (int c = 0; c < N_COMMON; ++c) common_probs_row[static_cast<size_t>(c)] /= row_sum;
      result.common_probs.push_back(std::move(common_probs_row));
    } else {
      // No common taxonomy: use network class indices and raw softmax directly
      point.label = static_cast<uint32_t>(best_class);
      result.common_probs.push_back(softmax_net);
    }
    pc->points.push_back(point);
  }

  std::fclose(fp);
  pc->width = static_cast<uint32_t>(pc->points.size());
  return result;
}

pcl::PointCloud<pcl::PointXYZL>::Ptr CommonUtils::read_gt_scan_multiclass(
    const std::string& fn_scan, const std::string& fn_gt_multiclass) {
  pcl::PointCloud<pcl::PointXYZL>::Ptr pc(new pcl::PointCloud<pcl::PointXYZL>);
  FILE* fp = std::fopen(fn_scan.c_str(), "rb");
  if (!fp) {
    RCLCPP_WARN_STREAM(node_->get_logger(), "Cannot open scan file: " << fn_scan);
    return pc;
  }
  FILE* fp_gt = std::fopen(fn_gt_multiclass.c_str(), "rb");
  if (!fp_gt) {
    RCLCPP_WARN_STREAM(node_->get_logger(), "Cannot open GT multiclass file: " << fn_gt_multiclass);
    std::fclose(fp);
    return pc;
  }
  std::fseek(fp, 0L, SEEK_END);
  size_t scan_sz = std::ftell(fp);
  std::rewind(fp);
  int n_points = static_cast<int>(scan_sz / (sizeof(float) * 4));
  std::fseek(fp_gt, 0L, SEEK_END);
  size_t gt_sz = std::ftell(fp_gt);
  std::rewind(fp_gt);
  int n_mc = static_cast<int>(gt_sz / sizeof(uint16_t));
  if (n_points == 0 || n_mc == 0) {
    std::fclose(fp);
    std::fclose(fp_gt);
    return pc;
  }
  int n_classes = n_mc / n_points;
  if (n_mc != n_points * n_classes) {
    RCLCPP_ERROR_STREAM(node_->get_logger(), "GT multiclass file size mismatch");
    std::fclose(fp);
    std::fclose(fp_gt);
    return pc;
  }
  std::vector<uint16_t> mc_raw(static_cast<size_t>(n_mc));
  if (std::fread(mc_raw.data(), sizeof(uint16_t), n_mc, fp_gt) != static_cast<size_t>(n_mc)) {
    std::fclose(fp);
    std::fclose(fp_gt);
    return pc;
  }
  std::fclose(fp_gt);
  pc->points.reserve(n_points);
  pc->width = n_points;
  pc->height = 1;
  pc->is_dense = false;
  for (int i = 0; i < n_points; i++) {
    pcl::PointXYZL point;
    float intensity;
    if (std::fread(&point.x, sizeof(float), 1, fp) != 1) break;
    if (std::fread(&point.y, sizeof(float), 1, fp) != 1) break;
    if (std::fread(&point.z, sizeof(float), 1, fp) != 1) break;
    if (std::fread(&intensity, sizeof(float), 1, fp) != 1) break;
    const uint16_t* row = mc_raw.data() + static_cast<size_t>(i) * n_classes;
    int best_class = 0;
    float best_val = half_to_float(row[0]);
    for (int c = 1; c < n_classes; c++) {
      float v = half_to_float(row[c]);
      if (v > best_val) { best_val = v; best_class = c; }
    }
    int raw_label = best_class;
    if (!gt_learning_map_inv_.empty()) {
      auto it = gt_learning_map_inv_.find(best_class);
      if (it != gt_learning_map_inv_.end()) raw_label = it->second;
    }
    int common_label = raw_label;
    if (common_label_config_loaded_) {
      auto it_c = gt_to_common_.find(raw_label);
      if (it_c != gt_to_common_.end()) common_label = it_c->second;
      else common_label = 0;
    }
    point.label = static_cast<uint32_t>(common_label);
    pc->points.push_back(point);
  }
  std::fclose(fp);
  pc->width = static_cast<uint32_t>(pc->points.size());
  return pc;
}

std::vector<uint32_t> CommonUtils::read_gt_labels(const std::string& dir, int scan_file_num) {
  char buf[256];
  std::snprintf(buf, sizeof(buf), "%010d", scan_file_num);
  std::string path = dir + "/" + std::string(buf) + ".bin";
  FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) return {};
  std::fseek(fp, 0L, SEEK_END);
  size_t sz = std::ftell(fp);
  std::rewind(fp);
  int n = static_cast<int>(sz / sizeof(uint32_t));
  std::vector<uint32_t> labels(n);
  if (std::fread(labels.data(), sizeof(uint32_t), n, fp) != static_cast<size_t>(n))
    labels.clear();
  std::fclose(fp);
  return labels;
}
