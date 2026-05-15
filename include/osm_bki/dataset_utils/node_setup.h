#pragma once

#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "common_utils.h"
#include "markerarray_pub.h"

namespace osm_bki {
namespace node_setup {

// ---------------------------------------------------------------------------
// All ROS parameters the mcd_node / kitti360_node pipelines read.
// Per-dataset defaults that need to vary at declare-time (e.g. label keys)
// are passed as args to declare_parameters().
// ---------------------------------------------------------------------------
struct PipelineParams {
  // Mapping / BKI
  std::string map_topic{"/occupied_cells_vis_array"};
  int block_depth{4};
  double sf2{1.0};
  double ell{1.0};
  float prior{1.0f};
  float var_thresh{1.0f};
  double free_thresh{0.3};
  double occupied_thresh{0.7};
  double resolution{0.1};
  int num_class{2};
  double free_resolution{0.5};
  double ds_resolution{0.1};
  int scan_num{0};
  double max_range{-1};
  double keyframe_dist{0.0};

  // Dataset paths (after sequence_name + suffix resolution)
  std::string dir;
  std::string sequence_name;  // empty for mcd flows; kitti360's load_osm_geometries uses it
  std::string input_data_prefix;
  std::string input_label_prefix;
  std::string lidar_pose_file;
  std::string gt_label_prefix;
  std::string evaluation_result_prefix;
  bool query{false};
  bool publish_semantic_occ_map{false};

  // Config / calibration
  std::string colors_file;
  std::string calibration_file;
  std::string config_datasets_dir;

  // OSM source
  std::string osm_file;
  double osm_origin_lat{0.0};
  double osm_origin_lon{0.0};
  double osm_decay_meters{2.0};
  double osm_tree_point_radius_meters{5.0};

  // Multiclass / labels
  bool inferred_use_multiclass{false};
  bool gt_use_multiclass{false};
  bool use_uncertainty_filter{false};
  std::string inferred_labels_key;  // declared default supplied by caller
  std::string gt_labels_key;        // declared default supplied by caller
  std::string confusion_matrix_file;
  std::string uncertainty_filter_mode{"confusion_matrix"};
  double uncertainty_drop_percent{10.0};
  double uncertainty_min_weight{0.1};
  bool use_common_taxonomy{true};

  // OSM confusion matrix / priors
  std::string osm_confusion_matrix_file;
  double osm_prior_strength{0.0};
  double osm_dirichlet_prior_strength{0.0};
  double osm_scan_radius_extension{1.2};

  // Height filter
  bool osm_height_filtering{false};
  double height_kernel_lambda{0.0};
  std::vector<double> height_kernel_dead_zone;
  bool height_kernel_redistribute{false};
  double height_kernel_gate{0.0};
  double sensor_mounting_height{0.0};
  std::vector<double> height_kernel_mu;
  std::vector<double> height_kernel_tau;

  // OSM prior map publishing
  bool publish_osm_prior_map{false};
  std::string osm_prior_map_color_mode{"osm_blend"};
  std::string osm_prior_map_topic{"/semantic_osm_prior_map"};
  double osm_prior_map_z{0.0};

  // OSM-converted map publishing
  bool publish_osm_converted_map{false};
  std::string osm_converted_map_topic{"/semantic_osm_converted_map"};

  // Variance / semantic uncertainty publishing
  bool publish_variance{false};
  std::string variance_topic{"/osm_bki_variance"};
  bool publish_semantic_uncertainty{false};
  std::string semantic_uncertainty_topic{"/semantic_uncertainty_cloud"};

  // MCD-only flags (declared for all nodes; only mcd_node consults them)
  bool publish_static_tf{true};
  bool use_pose_index_as_scan_id{false};
};

/// Declare every ROS parameter the pipeline reads, with defaults.
/// `default_labels_key` becomes the default for both `inferred_labels_key` and
/// `gt_labels_key` (e.g. "mcd" or "kitti360").
void declare_parameters(rclcpp::Node::SharedPtr node,
                        const std::string& default_labels_key);

/// Read every parameter into a fresh PipelineParams. Also applies the
/// sequence_name + suffix path-building convention.
PipelineParams load_parameters(rclcpp::Node::SharedPtr node);

/// Resolve the dataset-config directory. If `override_dir` is non-empty, returns
/// it (with a trailing slash); else returns the installed package share dir.
std::string resolve_config_datasets_dir(const std::string& override_dir);

/// Map a labels_key ("mcd", "kitti360", anything else) to its label config file
/// inside `cfg_dir`.
std::string resolve_label_config_path(const std::string& labels_key,
                                      const std::string& cfg_dir);

/// Parse the OSM prior-map color mode string. Unknown values fall back to OSMBlend.
MapColorMode parse_osm_color_mode(const std::string& s);

/// Load body-to-lidar calibration. Returns false (and logs) on failure.
bool setup_calibration(CommonUtils& common_utils, rclcpp::Node::SharedPtr node);

/// Load common taxonomy + multiclass + uncertainty filter + GT multiclass label configs.
/// Returns false only on hard failures (e.g. missing common label config when required).
bool setup_label_configs(CommonUtils& common_utils,
                         rclcpp::Node::SharedPtr node,
                         const PipelineParams& p,
                         const std::string& cfg_dir);

/// Load the semantic colors yaml if `p.colors_file` is set.
void setup_colors(CommonUtils& common_utils,
                  rclcpp::Node::SharedPtr node,
                  const PipelineParams& p,
                  const std::string& cfg_dir);

/// Wire up variance/uncertainty/prior-map/converted-map publishing, OSM confusion
/// matrix, and (optionally) the height kernel.
void configure_osm_priors_and_publishing(CommonUtils& common_utils,
                                         rclcpp::Node::SharedPtr node,
                                         const PipelineParams& p,
                                         const std::string& cfg_dir);

}  // namespace node_setup
}  // namespace osm_bki
