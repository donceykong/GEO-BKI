#pragma once

#include <string>
#include <rclcpp/rclcpp.hpp>

#include "common_utils.h"

namespace osm_bki {
namespace kitti360 {

/// Parse a KITTI-360 velodyne_poses.txt file.
///
/// Each line: frame_index (int) followed by 12 or 16 floats (3x4 or 4x4 row-major).
/// The first pose's translation is stored in `out.original_first_pose` (translation
/// only, identity rotation — matching the Python visualize_sem_map_KITTI360.py
/// convention `trans = [-first_x, -first_y, 0]`), then subtracted from all poses
/// (rotation preserved).
///
/// @return true on success; false if the file can't be opened or contains no poses.
bool read_lidar_poses(const std::string& path,
                      rclcpp::Logger logger,
                      PoseData& out);

}  // namespace kitti360
}  // namespace osm_bki
