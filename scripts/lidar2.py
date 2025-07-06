import carla
import open3d as o3d
import numpy as np
import time

def visualize_lidar(points):
    # Ubah ke format (N, 3)
    xyz = np.array(points).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])

def main():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('points_per_second', '56000')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('range', '50')
    lidar_location = carla.Location(x=0, z=2.5)
    lidar_transform = carla.Transform(lidar_location)
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

    points = []

    def lidar_callback(point_cloud):
        array = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.float32))
        array = np.reshape(array, (-1, 4))  # x, y, z, intensity
        if len(points) == 0:
            points.extend(array[:, :3])  # Simpan hanya satu frame untuk demo

    lidar_sensor.listen(lambda data: lidar_callback(data))

    print("Mengumpulkan 1 frame Lidar...")
    time.sleep(2)  # Tunggu 2 detik untuk dapat data
    
    lidar_sensor.stop()
    lidar_sensor.destroy()
    vehicle.destroy()

    if len(points) > 0:
        print("Menampilkan visualisasi point cloud.")
        visualize_lidar(points)
    else:
        print("Belum ada data Lidar yang diterima.")

if __name__ == "__main__":
    main()