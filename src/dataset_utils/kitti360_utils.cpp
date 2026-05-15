#include "kitti360_utils.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace osm_bki {
namespace kitti360 {

bool read_lidar_poses(const std::string& path,
                      rclcpp::Logger logger,
                      PoseData& out) {
  std::ifstream f(path);
  if (!f.is_open()) {
    RCLCPP_ERROR_STREAM(logger, "Cannot open KITTI360 pose file: " << path);
    return false;
  }
  out.lidar_poses.clear();
  out.scan_indices.clear();
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    std::vector<double> values;
    int frame_index = -1;
    if (!(ss >> frame_index)) continue;
    double v;
    while (ss >> v) values.push_back(v);
    if (values.size() == 12u) {
      Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
      T.block<3, 4>(0, 0) = Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(values.data());
      out.lidar_poses.push_back(T);
      out.scan_indices.push_back(frame_index);
    } else if (values.size() == 16u) {
      Eigen::Matrix4d T = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(values.data());
      out.lidar_poses.push_back(T);
      out.scan_indices.push_back(frame_index);
    }
  }
  f.close();
  if (out.lidar_poses.empty()) {
    RCLCPP_ERROR_STREAM(logger, "No valid poses in " << path);
    return false;
  }

  // Store first pose as translation-only (Identity + first translation) so that
  // transformToFirstPoseOrigin produces a pure translation shift for OSM,
  // matching the Python: trans = [-first_x, -first_y, 0]
  Eigen::Vector3d first_t = out.lidar_poses[0].block<3, 1>(0, 3);
  out.original_first_pose = Eigen::Matrix4d::Identity();
  out.original_first_pose.block<3, 1>(0, 3) = first_t;

  // Subtract first pose translation from all poses (keep rotation intact)
  for (size_t i = 0; i < out.lidar_poses.size(); ++i)
    out.lidar_poses[i].block<3, 1>(0, 3) -= first_t;
  RCLCPP_INFO_STREAM(logger, "Loaded " << out.lidar_poses.size()
      << " KITTI360 poses from " << path
      << " (shifted by first translation [" << first_t.x() << ", " << first_t.y() << ", " << first_t.z() << "])");
  return true;
}

}  // namespace kitti360
}  // namespace osm_bki
