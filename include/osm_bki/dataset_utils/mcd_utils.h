#pragma once

#include <string>

#include <rclcpp/rclcpp.hpp>

#include "common_utils.h"

namespace osm_bki {
namespace mcd {

/// Parse an MCD CSV pose file (columns: num, timestamp, x, y, z, qx, qy, qz, qw).
///
/// Captures the original first pose into `out.original_first_pose`, then normalizes
/// all poses so the first pose becomes the identity (transforms by the inverse of
/// the first pose). Scan indices come from the first CSV column.
///
/// @return true on success; false if the file can't be opened or contains no poses.
bool read_lidar_poses(const std::string& path,
                      rclcpp::Logger logger,
                      PoseData& out);

/// Overwrite scan indices with 0, 1, 2, ... in pose order. Use when the pose file's
/// first column doesn't match lidar filenames (e.g. cu_north_campus poses.csv has
/// scan #s 4,5,6... but lidar bins are named 0000000000.bin, 0000000001.bin, ...).
void apply_pose_index_as_scan_id(PoseData& data, rclcpp::Logger logger);

}  // namespace mcd
}  // namespace osm_bki
