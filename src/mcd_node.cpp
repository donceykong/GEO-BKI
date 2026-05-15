#include <string>
#include <memory>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include "common_utils.h"
#include "mcd_utils.h"
#include "node_setup.h"
#include "osm_visualizer.h"

namespace {

using osm_bki::node_setup::PipelineParams;

void log_parameters(rclcpp::Node::SharedPtr node, const PipelineParams& p) {
  RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: All parameters retrieved. dir=" << p.dir
      << ", lidar_pose_file=" << p.lidar_pose_file
      << ", calibration_file=" << p.calibration_file);

  RCLCPP_INFO_STREAM(node->get_logger(), "Parameters:" << std::endl <<
    "block_depth: " << p.block_depth << std::endl <<
    "sf2: " << p.sf2 << std::endl <<
    "ell: " << p.ell << std::endl <<
    "prior:" << p.prior << std::endl <<
    "var_thresh: " << p.var_thresh << std::endl <<
    "free_thresh: " << p.free_thresh << std::endl <<
    "occupied_thresh: " << p.occupied_thresh << std::endl <<
    "resolution: " << p.resolution << std::endl <<
    "num_class: " << p.num_class << std::endl <<
    "free_resolution: " << p.free_resolution << std::endl <<
    "ds_resolution: " << p.ds_resolution << std::endl <<
    "scan_num: " << p.scan_num << std::endl <<
    "max_range: " << p.max_range << std::endl <<
    "keyframe_dist: " << p.keyframe_dist << std::endl <<

    "MCD:" << std::endl <<
    "dir: " << p.dir << std::endl <<
    "input_data_prefix: " << p.input_data_prefix << std::endl <<
    "input_label_prefix: " << p.input_label_prefix << std::endl <<
    "lidar_pose_file: " << p.lidar_pose_file << std::endl <<
    "gt_label_prefix: " << p.gt_label_prefix << std::endl <<
    "evaluation_result_prefix: " << p.evaluation_result_prefix << std::endl <<
    "query: " << p.query << std::endl <<
    "publish_semantic_occ_map: " << p.publish_semantic_occ_map
    );
}

void publish_static_map_to_odom_tf(rclcpp::Node::SharedPtr node) {
  try {
    tf2_ros::StaticTransformBroadcaster static_tf_broadcaster(node);
    geometry_msgs::msg::TransformStamped static_transform;
    static_transform.header.stamp = node->now();
    static_transform.header.frame_id = "map";
    static_transform.child_frame_id = "odom";
    static_transform.transform.translation.x = 0.0;
    static_transform.transform.translation.y = 0.0;
    static_transform.transform.translation.z = 0.0;
    static_transform.transform.rotation.x = 0.0;
    static_transform.transform.rotation.y = 0.0;
    static_transform.transform.rotation.z = 0.0;
    static_transform.transform.rotation.w = 1.0;
    static_tf_broadcaster.sendTransform(static_transform);
    RCLCPP_INFO(node->get_logger(), "Published static transform: map -> odom (identity)");
  } catch (const std::exception& e) {
    RCLCPP_WARN_STREAM(node->get_logger(), "WARNING: Exception publishing static transform: " << e.what());
  }
}

bool setup_poses(CommonUtils& common_utils,
                 rclcpp::Node::SharedPtr node,
                 const PipelineParams& p) {
  RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to read lidar poses from: "
      << (p.dir + '/' + p.lidar_pose_file));
  PoseData poses;
  if (!osm_bki::mcd::read_lidar_poses(p.dir + '/' + p.lidar_pose_file, node->get_logger(), poses)) {
    RCLCPP_WARN_STREAM(node->get_logger(), "WARNING: Failed to read lidar poses!");
    return false;
  }
  if (p.use_pose_index_as_scan_id)
    osm_bki::mcd::apply_pose_index_as_scan_id(poses, node->get_logger());
  common_utils.set_pose_data(std::move(poses));
  RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Lidar poses read successfully");
  return true;
}

void load_osm_geometries(CommonUtils& common_utils,
                         rclcpp::Node::SharedPtr node,
                         const PipelineParams& p) {
  if (p.osm_file.empty()) return;

  std::string full_osm_path = p.osm_file;
  if (p.osm_file[0] != '/' && !p.dir.empty()) {
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
    RCLCPP_WARN_STREAM(node->get_logger(), "Failed to load OSM file for priors: " << full_osm_path);
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
  RCLCPP_INFO_STREAM(node->get_logger(), "Loaded OSM geometries for voxel priors: "
      << osm_vis.getBuildings().size() << " buildings, " << osm_vis.getRoads().size() << " roads, "
      << osm_vis.getSidewalks().size() << " sidewalks, " << osm_vis.getCycleways().size() << " cycleways, "
      << osm_vis.getParking().size() << " parking, " << osm_vis.getFences().size() << " fences, "
      << osm_vis.getGrasslands().size() << " grasslands, " << osm_vis.getTrees().size() << " trees, "
      << osm_vis.getForests().size() << " forests, " << osm_vis.getTreePoints().size()
      << " tree points (decay=" << p.osm_decay_meters << " m)");
}

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("mcd_node");
  if (!node) {
    RCLCPP_WARN_STREAM(rclcpp::get_logger("mcd_node"), "WARNING: Failed to create ROS2 node!");
    return 1;
  }
  RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Node created successfully");

  osm_bki::node_setup::declare_parameters(node, "mcd");
  PipelineParams p = osm_bki::node_setup::load_parameters(node);
  log_parameters(node, p);

  if (p.publish_static_tf) {
    publish_static_map_to_odom_tf(node);
  } else {
    RCLCPP_INFO(node->get_logger(), "Skipping static TF (publish_static_tf=false).");
  }

  RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to create CommonUtils object");
  CommonUtils common_utils(node, p.resolution, p.block_depth, p.sf2, p.ell, p.num_class,
                           p.free_thresh, p.occupied_thresh, p.var_thresh,
                           p.ds_resolution, p.free_resolution, p.max_range,
                           p.map_topic, p.prior);
  RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: CommonUtils object created successfully");

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

  RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to set up evaluation");
  common_utils.set_up_evaluation(p.dir + '/' + p.gt_label_prefix, p.dir + '/' + p.evaluation_result_prefix);
  RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Evaluation setup completed");

  RCLCPP_WARN_STREAM(node->get_logger(),
      "CHECKPOINT: About to process scans. input_data_prefix=" << p.input_data_prefix
      << ", scan_num=" << p.scan_num);
  common_utils.process_scans(p.dir + '/' + p.input_data_prefix,
                             p.dir + '/' + p.input_label_prefix,
                             p.scan_num, p.keyframe_dist, p.query, p.publish_semantic_occ_map);
  RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Scan processing completed, about to spin");
  RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Starting rclcpp::spin(node)...");

  rclcpp::spin(node);

  RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: rclcpp::spin() returned (node shutdown)");
  rclcpp::shutdown();
  return 0;
}
