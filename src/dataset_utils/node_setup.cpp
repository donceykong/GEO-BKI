#include "node_setup.h"

#include <string>
#include <vector>

#include <ament_index_cpp/get_package_share_directory.hpp>

namespace osm_bki {
namespace node_setup {

void declare_parameters(rclcpp::Node::SharedPtr node,
                        const std::string& default_labels_key) {
  PipelineParams d;  // defaults
  node->declare_parameter<std::string>("map_topic", d.map_topic);
  node->declare_parameter<int>("block_depth", d.block_depth);
  node->declare_parameter<double>("sf2", d.sf2);
  node->declare_parameter<double>("ell", d.ell);
  node->declare_parameter<float>("prior", d.prior);
  node->declare_parameter<float>("var_thresh", d.var_thresh);
  node->declare_parameter<double>("free_thresh", d.free_thresh);
  node->declare_parameter<double>("occupied_thresh", d.occupied_thresh);
  node->declare_parameter<double>("resolution", d.resolution);
  node->declare_parameter<int>("num_class", d.num_class);
  node->declare_parameter<double>("free_resolution", d.free_resolution);
  node->declare_parameter<double>("ds_resolution", d.ds_resolution);
  node->declare_parameter<int>("scan_num", d.scan_num);
  node->declare_parameter<double>("max_range", d.max_range);
  node->declare_parameter<double>("keyframe_dist", d.keyframe_dist);
  node->declare_parameter<std::string>("dir", std::string{});
  node->declare_parameter<std::string>("sequence_name", "");
  node->declare_parameter<std::string>("input_data_suffix", "");
  node->declare_parameter<std::string>("input_label_suffix", "");
  node->declare_parameter<std::string>("lidar_pose_suffix", "");
  node->declare_parameter<std::string>("gt_label_suffix", "");
  node->declare_parameter<std::string>("input_data_prefix", std::string{});
  node->declare_parameter<std::string>("input_label_prefix", std::string{});
  node->declare_parameter<std::string>("lidar_pose_file", std::string{});
  node->declare_parameter<std::string>("gt_label_prefix", std::string{});
  node->declare_parameter<std::string>("evaluation_result_prefix", std::string{});
  node->declare_parameter<bool>("query", d.query);
  node->declare_parameter<bool>("publish_semantic_occ_map", d.publish_semantic_occ_map);
  node->declare_parameter<std::string>("colors_file", "");
  node->declare_parameter<std::string>("calibration_file", "");
  node->declare_parameter<std::string>("osm_file", "");
  node->declare_parameter<double>("osm_origin_lat", d.osm_origin_lat);
  node->declare_parameter<double>("osm_origin_lon", d.osm_origin_lon);
  node->declare_parameter<double>("osm_decay_meters", d.osm_decay_meters);
  node->declare_parameter<double>("osm_tree_point_radius_meters", d.osm_tree_point_radius_meters);
  node->declare_parameter<bool>("inferred_use_multiclass", d.inferred_use_multiclass);
  node->declare_parameter<bool>("gt_use_multiclass", d.gt_use_multiclass);
  node->declare_parameter<bool>("use_uncertainty_filter", d.use_uncertainty_filter);
  node->declare_parameter<std::string>("inferred_labels_key", default_labels_key);
  node->declare_parameter<std::string>("gt_labels_key", default_labels_key);
  node->declare_parameter<std::string>("confusion_matrix_file", "");
  node->declare_parameter<std::string>("uncertainty_filter_mode", d.uncertainty_filter_mode);
  node->declare_parameter<double>("uncertainty_drop_percent", d.uncertainty_drop_percent);
  node->declare_parameter<double>("uncertainty_min_weight", d.uncertainty_min_weight);
  node->declare_parameter<std::string>("config_datasets_dir", "");
  node->declare_parameter<std::string>("osm_confusion_matrix_file", "");
  node->declare_parameter<double>("osm_prior_strength", d.osm_prior_strength);
  node->declare_parameter<double>("osm_dirichlet_prior_strength", d.osm_dirichlet_prior_strength);
  node->declare_parameter<double>("osm_scan_radius_extension", d.osm_scan_radius_extension);
  node->declare_parameter<bool>("osm_height_filtering", d.osm_height_filtering);
  node->declare_parameter<double>("height_kernel_lambda", d.height_kernel_lambda);
  node->declare_parameter<std::vector<double>>("height_kernel_dead_zone", std::vector<double>{});
  node->declare_parameter<bool>("height_kernel_redistribute", d.height_kernel_redistribute);
  node->declare_parameter<double>("height_kernel_gate", d.height_kernel_gate);
  node->declare_parameter<double>("sensor_mounting_height", d.sensor_mounting_height);
  node->declare_parameter<std::vector<double>>("height_kernel_mu", std::vector<double>{});
  node->declare_parameter<std::vector<double>>("height_kernel_tau", std::vector<double>{});
  node->declare_parameter<bool>("publish_osm_prior_map", d.publish_osm_prior_map);
  node->declare_parameter<std::string>("osm_prior_map_color_mode", d.osm_prior_map_color_mode);
  node->declare_parameter<std::string>("osm_prior_map_topic", d.osm_prior_map_topic);
  node->declare_parameter<double>("osm_prior_map_z", d.osm_prior_map_z);
  node->declare_parameter<bool>("publish_osm_converted_map", d.publish_osm_converted_map);
  node->declare_parameter<std::string>("osm_converted_map_topic", d.osm_converted_map_topic);
  node->declare_parameter<bool>("publish_variance", d.publish_variance);
  node->declare_parameter<std::string>("variance_topic", d.variance_topic);
  node->declare_parameter<bool>("publish_semantic_uncertainty", d.publish_semantic_uncertainty);
  node->declare_parameter<std::string>("semantic_uncertainty_topic", d.semantic_uncertainty_topic);
  node->declare_parameter<bool>("publish_static_tf", d.publish_static_tf);
  node->declare_parameter<bool>("use_pose_index_as_scan_id", d.use_pose_index_as_scan_id);
  node->declare_parameter<bool>("use_common_taxonomy", d.use_common_taxonomy);
}

PipelineParams load_parameters(rclcpp::Node::SharedPtr node) {
  PipelineParams p;
  node->get_parameter<std::string>("map_topic", p.map_topic);
  node->get_parameter<int>("block_depth", p.block_depth);
  node->get_parameter<double>("sf2", p.sf2);
  node->get_parameter<double>("ell", p.ell);
  node->get_parameter<float>("prior", p.prior);
  node->get_parameter<float>("var_thresh", p.var_thresh);
  node->get_parameter<double>("free_thresh", p.free_thresh);
  node->get_parameter<double>("occupied_thresh", p.occupied_thresh);
  node->get_parameter<double>("resolution", p.resolution);
  node->get_parameter<int>("num_class", p.num_class);
  node->get_parameter<double>("free_resolution", p.free_resolution);
  node->get_parameter<double>("ds_resolution", p.ds_resolution);
  node->get_parameter<int>("scan_num", p.scan_num);
  node->get_parameter<double>("max_range", p.max_range);
  node->get_parameter<double>("keyframe_dist", p.keyframe_dist);
  node->get_parameter<std::string>("dir", p.dir);

  std::string input_data_suffix, input_label_suffix, lidar_pose_suffix, gt_label_suffix;
  node->get_parameter<std::string>("sequence_name", p.sequence_name);
  node->get_parameter<std::string>("input_data_suffix", input_data_suffix);
  node->get_parameter<std::string>("input_label_suffix", input_label_suffix);
  node->get_parameter<std::string>("lidar_pose_suffix", lidar_pose_suffix);
  node->get_parameter<std::string>("gt_label_suffix", gt_label_suffix);
  node->get_parameter<std::string>("input_data_prefix", p.input_data_prefix);
  node->get_parameter<std::string>("input_label_prefix", p.input_label_prefix);
  node->get_parameter<std::string>("lidar_pose_file", p.lidar_pose_file);
  node->get_parameter<std::string>("gt_label_prefix", p.gt_label_prefix);
  node->get_parameter<std::string>("evaluation_result_prefix", p.evaluation_result_prefix);
  // Build paths from sequence_name + suffix when sequence-based config is used.
  if (!p.sequence_name.empty() && !input_data_suffix.empty()) {
    p.input_data_prefix = p.sequence_name + "/" + input_data_suffix;
    if (!lidar_pose_suffix.empty()) p.lidar_pose_file = p.sequence_name + "/" + lidar_pose_suffix;
    if (!input_label_suffix.empty()) p.input_label_prefix = p.sequence_name + "/" + input_label_suffix;
    if (!gt_label_suffix.empty()) p.gt_label_prefix = p.sequence_name + "/" + gt_label_suffix;
    if (!p.evaluation_result_prefix.empty()) p.evaluation_result_prefix = p.sequence_name + "/" + p.evaluation_result_prefix;
  }

  node->get_parameter<bool>("query", p.query);
  node->get_parameter<bool>("publish_semantic_occ_map", p.publish_semantic_occ_map);
  node->get_parameter<std::string>("colors_file", p.colors_file);
  node->get_parameter<std::string>("config_datasets_dir", p.config_datasets_dir);
  node->get_parameter<std::string>("calibration_file", p.calibration_file);
  node->get_parameter<std::string>("osm_file", p.osm_file);
  node->get_parameter<double>("osm_origin_lat", p.osm_origin_lat);
  node->get_parameter<double>("osm_origin_lon", p.osm_origin_lon);
  node->get_parameter<double>("osm_decay_meters", p.osm_decay_meters);
  node->get_parameter<double>("osm_tree_point_radius_meters", p.osm_tree_point_radius_meters);

  node->get_parameter<bool>("inferred_use_multiclass", p.inferred_use_multiclass);
  node->get_parameter<bool>("gt_use_multiclass", p.gt_use_multiclass);
  node->get_parameter<bool>("use_uncertainty_filter", p.use_uncertainty_filter);
  node->get_parameter<std::string>("inferred_labels_key", p.inferred_labels_key);
  node->get_parameter<std::string>("gt_labels_key", p.gt_labels_key);
  node->get_parameter<std::string>("confusion_matrix_file", p.confusion_matrix_file);
  node->get_parameter<std::string>("uncertainty_filter_mode", p.uncertainty_filter_mode);
  node->get_parameter<double>("uncertainty_drop_percent", p.uncertainty_drop_percent);
  node->get_parameter<double>("uncertainty_min_weight", p.uncertainty_min_weight);
  node->get_parameter<bool>("use_common_taxonomy", p.use_common_taxonomy);

  node->get_parameter<std::string>("osm_confusion_matrix_file", p.osm_confusion_matrix_file);
  node->get_parameter<double>("osm_prior_strength", p.osm_prior_strength);
  node->get_parameter<double>("osm_dirichlet_prior_strength", p.osm_dirichlet_prior_strength);
  node->get_parameter<double>("osm_scan_radius_extension", p.osm_scan_radius_extension);

  node->get_parameter<bool>("osm_height_filtering", p.osm_height_filtering);
  node->get_parameter<double>("height_kernel_lambda", p.height_kernel_lambda);
  node->get_parameter<std::vector<double>>("height_kernel_dead_zone", p.height_kernel_dead_zone);
  node->get_parameter<bool>("height_kernel_redistribute", p.height_kernel_redistribute);
  node->get_parameter<double>("height_kernel_gate", p.height_kernel_gate);
  node->get_parameter<double>("sensor_mounting_height", p.sensor_mounting_height);
  node->get_parameter<std::vector<double>>("height_kernel_mu", p.height_kernel_mu);
  node->get_parameter<std::vector<double>>("height_kernel_tau", p.height_kernel_tau);

  node->get_parameter<bool>("publish_osm_prior_map", p.publish_osm_prior_map);
  node->get_parameter<std::string>("osm_prior_map_color_mode", p.osm_prior_map_color_mode);
  node->get_parameter<std::string>("osm_prior_map_topic", p.osm_prior_map_topic);
  node->get_parameter<double>("osm_prior_map_z", p.osm_prior_map_z);

  node->get_parameter<bool>("publish_osm_converted_map", p.publish_osm_converted_map);
  node->get_parameter<std::string>("osm_converted_map_topic", p.osm_converted_map_topic);

  node->get_parameter<bool>("publish_variance", p.publish_variance);
  node->get_parameter<std::string>("variance_topic", p.variance_topic);
  node->get_parameter<bool>("publish_semantic_uncertainty", p.publish_semantic_uncertainty);
  node->get_parameter<std::string>("semantic_uncertainty_topic", p.semantic_uncertainty_topic);

  node->get_parameter<bool>("publish_static_tf", p.publish_static_tf);
  node->get_parameter<bool>("use_pose_index_as_scan_id", p.use_pose_index_as_scan_id);
  return p;
}

std::string resolve_config_datasets_dir(const std::string& override_dir) {
  if (!override_dir.empty()) {
    return override_dir.back() == '/' ? override_dir : override_dir + '/';
  }
  return ament_index_cpp::get_package_share_directory("osm_bki") + "/config/datasets/";
}

std::string resolve_label_config_path(const std::string& labels_key,
                                      const std::string& cfg_dir) {
  std::string f;
  if (labels_key == "mcd") f = "labels_mcd.yaml";
  else if (labels_key == "kitti360") f = "labels_kitti360.yaml";
  else f = "labels_semkitti.yaml";
  return cfg_dir + f;
}

MapColorMode parse_osm_color_mode(const std::string& s) {
  if (s == "osm_building")  return MapColorMode::OSMBuilding;
  if (s == "osm_road")      return MapColorMode::OSMRoad;
  if (s == "osm_grassland") return MapColorMode::OSMGrassland;
  if (s == "osm_tree")      return MapColorMode::OSMTree;
  if (s == "osm_parking")   return MapColorMode::OSMParking;
  if (s == "osm_fence")     return MapColorMode::OSMFence;
  if (s == "osm_sidewalk")  return MapColorMode::OSMSidewalk;
  if (s == "osm_cycleway")  return MapColorMode::OSMCycleway;
  if (s == "osm_forest")    return MapColorMode::OSMForest;
  return MapColorMode::OSMBlend;
}

bool setup_calibration(CommonUtils& common_utils, rclcpp::Node::SharedPtr node) {
  if (!common_utils.load_calibration_from_params()) {
    RCLCPP_FATAL(node->get_logger(),
        "Failed to load body-to-lidar calibration! Cannot proceed without calibration.");
    return false;
  }
  return true;
}

bool setup_label_configs(CommonUtils& common_utils,
                         rclcpp::Node::SharedPtr node,
                         const PipelineParams& p,
                         const std::string& cfg_dir) {
  if (p.use_common_taxonomy) {
    std::string common_label_path = cfg_dir + "labels_common.yaml";
    if (!common_utils.load_common_label_config(common_label_path, p.inferred_labels_key, p.gt_labels_key)) {
      RCLCPP_FATAL_STREAM(node->get_logger(),
          "Failed to load common label config from " << common_label_path
          << ". Cannot proceed without label mappings.");
      return false;
    }
  } else {
    RCLCPP_INFO(node->get_logger(),
        "use_common_taxonomy=false: using network class indices (set num_class to network n_classes)");
  }

  if (p.inferred_use_multiclass) {
    common_utils.set_inferred_multiclass_mode(true, p.dir + '/' + p.input_label_prefix);
    if (!common_utils.load_label_config(resolve_label_config_path(p.inferred_labels_key, cfg_dir))) {
      RCLCPP_WARN_STREAM(node->get_logger(),
          "Failed to load inferred label config. Argmax indices will be used as-is.");
    }
    common_utils.set_uncertainty_filter(p.use_uncertainty_filter, p.inferred_labels_key,
                                        p.uncertainty_filter_mode,
                                        static_cast<float>(p.uncertainty_drop_percent),
                                        static_cast<float>(p.uncertainty_min_weight));
    if (p.use_uncertainty_filter && !p.confusion_matrix_file.empty()) {
      std::string cm_path = cfg_dir + p.confusion_matrix_file;
      if (!common_utils.load_confusion_matrix(cm_path)) {
        RCLCPP_WARN_STREAM(node->get_logger(),
            "Failed to load confusion matrix. Uncertainty filtering disabled.");
      }
    }
  }

  if (p.gt_use_multiclass) {
    common_utils.set_gt_multiclass_mode(true);
    if (!common_utils.load_gt_label_config(resolve_label_config_path(p.gt_labels_key, cfg_dir))) {
      RCLCPP_WARN_STREAM(node->get_logger(),
          "Failed to load GT label config. Argmax indices will be used as-is.");
    }
  }
  return true;
}

void setup_colors(CommonUtils& common_utils,
                  rclcpp::Node::SharedPtr node,
                  const PipelineParams& p,
                  const std::string& cfg_dir) {
  if (p.colors_file.empty()) {
    RCLCPP_INFO(node->get_logger(),
        "No colors_file specified in dataset config. Using default hardcoded colors.");
    return;
  }
  std::string colors_file_path = cfg_dir + p.colors_file;
  if (!common_utils.load_colors_from_yaml(colors_file_path)) {
    RCLCPP_WARN_STREAM(node->get_logger(),
        "Failed to load colors from: " << colors_file_path
        << ". Using default hardcoded colors.");
  }
}

void configure_osm_priors_and_publishing(CommonUtils& common_utils,
                                         rclcpp::Node::SharedPtr node,
                                         const PipelineParams& p,
                                         const std::string& cfg_dir) {
  common_utils.set_publish_variance(p.publish_variance, p.variance_topic);
  common_utils.set_publish_semantic_uncertainty(p.publish_semantic_uncertainty, p.semantic_uncertainty_topic);
  common_utils.set_osm_prior_strength(static_cast<float>(p.osm_prior_strength));
  common_utils.set_osm_dirichlet_prior_strength(static_cast<float>(p.osm_dirichlet_prior_strength));
  common_utils.set_osm_scan_radius_extension(static_cast<float>(p.osm_scan_radius_extension));

  if (p.osm_height_filtering) {
    std::vector<float> mu_f(p.height_kernel_mu.begin(), p.height_kernel_mu.end());
    std::vector<float> tau_f(p.height_kernel_tau.begin(), p.height_kernel_tau.end());
    std::vector<float> dz_f(p.height_kernel_dead_zone.begin(), p.height_kernel_dead_zone.end());
    common_utils.set_height_kernel_params(static_cast<float>(p.height_kernel_lambda), mu_f, tau_f,
                                          dz_f, p.height_kernel_redistribute,
                                          static_cast<float>(p.height_kernel_gate),
                                          static_cast<float>(p.sensor_mounting_height));
    RCLCPP_INFO_STREAM(node->get_logger(),
        "Height filter: gaussian (lambda=" << p.height_kernel_lambda
        << ", dead_zone=[" << dz_f.size() << " per-class]"
        << ", gate=" << p.height_kernel_gate
        << ", sensor_height=" << p.sensor_mounting_height
        << ", " << mu_f.size() << " mu / " << tau_f.size() << " tau)");
  }

  common_utils.set_publish_osm_prior_map(p.publish_osm_prior_map, p.osm_prior_map_topic,
                                         parse_osm_color_mode(p.osm_prior_map_color_mode),
                                         static_cast<float>(p.osm_prior_map_z));
  common_utils.set_publish_osm_converted_map(p.publish_osm_converted_map, p.osm_converted_map_topic);

  // Widths and decay come from ROS params (osm_geometry_parameters.*) applied before
  // loadFromOSM; do not re-load from the confusion-matrix yaml here.
  if (!p.osm_confusion_matrix_file.empty() && p.osm_prior_strength > 0.0) {
    std::string cm_path = cfg_dir + p.osm_confusion_matrix_file;
    if (common_utils.load_osm_confusion_matrix(cm_path)) {
      RCLCPP_INFO_STREAM(node->get_logger(),
          "Loaded OSM confusion matrix from " << cm_path
          << " (strength=" << p.osm_prior_strength << ")");
    } else {
      RCLCPP_WARN_STREAM(node->get_logger(),
          "Failed to load OSM confusion matrix from " << cm_path);
    }
  }
}

}  // namespace node_setup
}  // namespace osm_bki
