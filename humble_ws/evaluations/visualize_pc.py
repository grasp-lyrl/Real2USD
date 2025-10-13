import open3d as o3d

# pcd = o3d.io.read_point_cloud("evaluations/pc_data/hallway-0_voxel_pointcloud.ply")
pcd = o3d.io.read_point_cloud("evaluations/pc_data/hallway1-half_voxel_pointcloud.ply")
# pcd = o3d.io.read_point_cloud("evaluations/pc_data/smalloffice-0_voxel_pointcloud.ply")
# pcd = o3d.io.read_point_cloud("evaluations/pc_data/smalloffice-1_voxel_pointcloud.ply")
# pcd = o3d.io.read_point_cloud("evaluations/pc_data/lounge-0_voxel_pointcloud.ply")
o3d.visualization.draw_geometries([pcd])