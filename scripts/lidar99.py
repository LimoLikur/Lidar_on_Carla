import carla
import open3d as o3d
import numpy as np
import time
import random

latest_points = None

def lidar_callback(point_cloud):
    global latest_points
    points = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
    print("Jumlah titik lidar:", points.shape[0])
    if points.shape[0] > 0:
        print("Contoh point:", points[0])
    latest_points = points

def main():
    global latest_points
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle')[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '64')
    lidar_bp.set_attribute('points_per_second', '130000')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('range', '100')
    lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=2.5))
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

    lidar_sensor.listen(lidar_callback)

    print("Menunggu data Lidar...")
    while latest_points is None or len(latest_points) == 0:
        time.sleep(0.1)
    print("Data Lidar didapat, membuka window Open3D...")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Lidar Point Cloud")
    render_option = vis.get_render_option()
    render_option.point_size = 7.0
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    try:
        while True:
            if latest_points is not None and len(latest_points) > 0:
                pcd.points = o3d.utility.Vector3dVector(latest_points)
                pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (latest_points.shape[0], 1)))  # warna biru
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Keluar...")
    finally:
        vis.destroy_window()
        lidar_sensor.stop()
        lidar_sensor.destroy()
        vehicle.destroy()

if __name__ == "__main__":
    main()