#include "mcd_utils.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

namespace osm_bki {
namespace mcd {

bool read_lidar_poses(const std::string& path,
                      rclcpp::Logger logger,
                      PoseData& out) {
  RCLCPP_WARN_STREAM(logger, "CHECKPOINT: read_lidar_poses: Opening file: " << path);
  std::ifstream fPoses;
  fPoses.open(path.c_str());
  if (!fPoses.is_open()) {
    RCLCPP_WARN_STREAM(logger, "WARNING: Cannot open pose file " << path);
    RCLCPP_ERROR_STREAM(logger, "Cannot open pose file " << path);
    return false;
  }
  RCLCPP_WARN_STREAM(logger, "CHECKPOINT: Pose file opened successfully");

  // Skip header line if present
  std::string header_line;
  std::getline(fPoses, header_line);

  // Check if header contains column names (common in CSV)
  bool has_header = (header_line.find("num") != std::string::npos ||
                     header_line.find("timestamp") != std::string::npos ||
                     header_line.find("x") != std::string::npos);

  if (!has_header) {
    // No header, rewind to beginning
    fPoses.close();
    fPoses.open(path.c_str());
  }

  while (!fPoses.eof()) {
    std::string s;
    std::getline(fPoses, s);

    // Skip empty lines and comments
    if (s.empty() || s[0] == '#') {
      continue;
    }

    std::stringstream ss(s);
    std::string token;
    std::vector<double> values;

    // Parse CSV line (handles comma-separated or space-separated)
    char delimiter = ',';
    if (s.find(',') == std::string::npos) {
      delimiter = ' ';
    }

    while (std::getline(ss, token, delimiter)) {
      try {
        double val = std::stod(token);
        values.push_back(val);
      } catch (...) {
        // Skip invalid tokens
        continue;
      }
    }

    // Expect at least 8 values: num, timestamp, x, y, z, qx, qy, qz, qw
    // Or 9 values if first is index
    if (values.size() < 8) {
      continue;
    }

    // Extract scan index (num) - first column in CSV
    int scan_index = static_cast<int>(values[0]);

    // Extract pose values (skip num and timestamp, they're at indices 0 and 1)
    double x = values[2];
    double y = values[3];
    double z = values[4];
    double qx = values[5];
    double qy = values[6];
    double qz = values[7];
    double qw = values.size() > 8 ? values[8] : 1.0;  // Default qw to 1.0 if not provided

    // Convert quaternion to rotation matrix
    Eigen::Quaterniond quat(qw, qx, qy, qz);
    quat.normalize();

    // Build transformation matrix
    Eigen::Matrix4d t_matrix = Eigen::Matrix4d::Identity();
    t_matrix.block<3, 3>(0, 0) = quat.toRotationMatrix();
    t_matrix(0, 3) = x;
    t_matrix(1, 3) = y;
    t_matrix(2, 3) = z;

    out.lidar_poses.push_back(t_matrix);
    out.scan_indices.push_back(scan_index);
  }

  fPoses.close();
  RCLCPP_WARN_STREAM(logger, "CHECKPOINT: Finished reading pose file, loaded " << out.lidar_poses.size() << " poses");

  if (out.lidar_poses.empty()) {
    RCLCPP_WARN_STREAM(logger, "WARNING: No poses loaded from " << path);
    RCLCPP_ERROR_STREAM(logger, "No poses loaded from " << path);
    return false;
  }

  // Store original first pose before transformation (needed for OSM data alignment)
  out.original_first_pose = out.lidar_poses[0];

  // Make all poses relative to the first pose (set first pose to origin)
  Eigen::Matrix4d first_pose_inverse = out.lidar_poses[0].inverse();

  RCLCPP_INFO_STREAM(logger, "First pose before alignment:");
  RCLCPP_INFO_STREAM(logger, "  Translation: [" << out.lidar_poses[0](0,3) << ", " << out.lidar_poses[0](1,3) << ", " << out.lidar_poses[0](2,3) << "]");

  for (size_t i = 0; i < out.lidar_poses.size(); ++i) {
    out.lidar_poses[i] = first_pose_inverse * out.lidar_poses[i];
  }

  RCLCPP_INFO_STREAM(logger, "After alignment - First pose should be identity:");
  RCLCPP_INFO_STREAM(logger, "  Translation: [" << out.lidar_poses[0](0,3) << ", " << out.lidar_poses[0](1,3) << ", " << out.lidar_poses[0](2,3) << "]");
  RCLCPP_INFO_STREAM(logger, "Loaded " << out.lidar_poses.size() << " poses from " << path << " (all relative to first pose)");

  return true;
}

void apply_pose_index_as_scan_id(PoseData& data, rclcpp::Logger logger) {
  for (size_t i = 0; i < data.scan_indices.size(); ++i)
    data.scan_indices[i] = static_cast<int>(i);
  RCLCPP_INFO_STREAM(logger, "Applied pose index as scan ID: scan_indices are now 0.." << (data.scan_indices.size() - 1) << " (expect 0000000000.bin, 0000000001.bin, ... in pose order)");
}

}  // namespace mcd
}  // namespace osm_bki
