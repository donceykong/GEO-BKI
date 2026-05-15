#include <string>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "common_utils.h"
#include "kitti360_utils.h"
#include "node_setup.h"
#include "osm_visualizer.h"

namespace {

using osm_bki::node_setup::PipelineParams;

bool setup_poses(CommonUtils& common_utils,
                 rclcpp::Node::SharedPtr node,
                 const PipelineParams& p) {
  std::string pose_path = p.dir + "/" + p.lidar_pose_file;
  RCLCPP_INFO_STREAM(node->get_logger(), "Reading KITTI360 poses from: " << pose_path);
  PoseData poses;
  if (!osm_bki::kitti360::read_lidar_poses(pose_path, node->get_logger(), poses)) {
    RCLCPP_ERROR_STREAM(node->get_logger(), "Failed to read KITTI360 poses!");
    return false;
  }
  common_utils.set_pose_data(std::move(poses));
  return true;
}

void load_osm_geometries(CommonUtils& common_utils,
                         rclcpp::Node::SharedPtr node,
                         const PipelineParams& p) {
  if (p.osm_file.empty()) return;

  std::string full_osm_path = p.osm_file;
  if (p.osm_file[0] != '/' && !p.dir.empty()) {
    if (!p.sequence_name.empty())
      full_osm_path = p.dir + "/" + p.sequence_name + "/" + p.osm_file;
    else
      full_osm_path = p.dir + "/" + p.osm_file;
  }
  osm_bki::OSMVisualizer osm_vis(node, "");

  // Apply OSM geometry widths from ROS params (from osm_bki.yaml via launch, flattened as
  // "osm_geometry_parameters.<name>") BEFORE loadFromOSM so RawNetLines pick up the
  // configured widths rather than the hardcoded OSMVisualizer defaults.
  auto apply_width = [&](const std::string& name, auto setter) {
    if (!node->has_parameter(name)) {
      node->declare_parameter<double>(name, 0.0);
    }
    double v = 0.0;
    node->get_parameter(name, v);
    if (v > 0.0) setter(static_cast<float>(v));
  };
  apply_width("osm_geometry_parameters.road_width_meters",     [&](float w){ osm_vis.setRoadWidth(w); });
  apply_width("osm_geometry_parameters.sidewalk_width_meters", [&](float w){ osm_vis.setSidewalkWidth(w); });
  apply_width("osm_geometry_parameters.cycleway_width_meters", [&](float w){ osm_vis.setCyclewayWidth(w); });
  apply_width("osm_geometry_parameters.fence_width_meters",    [&](float w){ osm_vis.setFenceWidth(w); });
  RCLCPP_INFO_STREAM(node->get_logger(), "OSM widths (m): road=" << osm_vis.getRoadWidth()
      << " sidewalk=" << osm_vis.getSidewalkWidth()
      << " cycleway=" << osm_vis.getCyclewayWidth()
      << " fence=" << osm_vis.getFenceWidth());

  if (!osm_vis.loadFromOSM(full_osm_path, p.osm_origin_lat, p.osm_origin_lon)) {
    RCLCPP_WARN_STREAM(node->get_logger(), "Failed to load OSM: " << full_osm_path);
    return;
  }

  osm_vis.transformToFirstPoseOrigin(common_utils.getOriginalFirstPose());
  common_utils.set_osm_buildings(osm_vis.getBuildings());
  common_utils.set_osm_roads(osm_vis.getRoads());
  common_utils.set_osm_sidewalks(osm_vis.getSidewalks());
  common_utils.set_osm_cycleways(osm_vis.getCycleways());
  common_utils.set_osm_grasslands(osm_vis.getGrasslands());
  common_utils.set_osm_trees(osm_vis.getTrees());
  common_utils.set_osm_forests(osm_vis.getForests());
  common_utils.set_osm_tree_points(osm_vis.getTreePoints());
  common_utils.set_osm_parking(osm_vis.getParking());
  common_utils.set_osm_fences(osm_vis.getFences());
  common_utils.set_osm_road_width(osm_vis.getRoadWidth());
  common_utils.set_osm_sidewalk_width(osm_vis.getSidewalkWidth());
  common_utils.set_osm_cycleway_width(osm_vis.getCyclewayWidth());
  common_utils.set_osm_fence_width(osm_vis.getFenceWidth());
  RCLCPP_INFO_STREAM(node->get_logger(), "Loaded OSM from " << full_osm_path);
}

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("kitti360_node");
  if (!node) {
    RCLCPP_ERROR_STREAM(rclcpp::get_logger("kitti360_node"), "Failed to create ROS2 node!");
    return 1;
  }

  osm_bki::node_setup::declare_parameters(node, "kitti360");
  PipelineParams p = osm_bki::node_setup::load_parameters(node);

  RCLCPP_INFO_STREAM(node->get_logger(), "KITTI360: dir=" << p.dir
      << ", lidar_pose_file=" << p.lidar_pose_file
      << ", calibration_file=" << (p.calibration_file.empty() ? "(identity)" : p.calibration_file));

  // No static TF for KITTI360 (poses are already in world frame, no base-to-lidar TF needed).

  CommonUtils common_utils(node, p.resolution, p.block_depth, p.sf2, p.ell, p.num_class,
                           p.free_thresh, p.occupied_thresh, p.var_thresh,
                           p.ds_resolution, p.free_resolution, p.max_range,
                           p.map_topic, p.prior);

  if (!setup_poses(common_utils, node, p)) return 1;
  if (!osm_bki::node_setup::setup_calibration(common_utils, node)) return 1;

  std::string cfg_dir = osm_bki::node_setup::resolve_config_datasets_dir(p.config_datasets_dir);
  if (!osm_bki::node_setup::setup_label_configs(common_utils, node, p, cfg_dir)) return 1;

  osm_bki::node_setup::setup_colors(common_utils, node, p, cfg_dir);
  common_utils.set_color_mode(osm_bki::MapColorMode::Semantic);
  common_utils.set_osm_decay_meters(static_cast<float>(p.osm_decay_meters));
  common_utils.set_osm_tree_point_radius(static_cast<float>(p.osm_tree_point_radius_meters));

  load_osm_geometries(common_utils, node, p);
  osm_bki::node_setup::configure_osm_priors_and_publishing(common_utils, node, p, cfg_dir);

  common_utils.set_up_evaluation(p.dir + "/" + p.gt_label_prefix, p.dir + "/" + p.evaluation_result_prefix);
  common_utils.process_scans(p.dir + "/" + p.input_data_prefix,
                             p.dir + "/" + p.input_label_prefix,
                             p.scan_num, p.keyframe_dist, p.query, p.publish_semantic_occ_map);

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
