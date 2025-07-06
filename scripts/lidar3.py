import carla
import open3d as o3d
import numpy as np
import cv2
import threading
import time

# Variabel global untuk data
lidar_points = []
camera_image = None
stop_flag = False

def lidar_callback(point_cloud):
    global lidar_points
    array = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
    array = np.reshape(array, (-1, 4))
    lidar_points = array[:, :3]

def camera_callback(image):
    global camera_image
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))
    camera_image = array[:, :, :3][:, :, ::-1]  # BGRA to RGB

def visualize_lidar():
    global stop_flag
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Lidar Point Cloud')
    pcd = o3d.geometry.PointCloud()
    added = False
    while not stop_flag:
        if len(lidar_points) > 0:
            pcd.points = o3d.utility.Vector3dVector(np.copy(lidar_points))
            if not added:
                vis.add_geometry(pcd)
                added = True
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
        time.sleep(0.05)
    vis.destroy_window()

def visualize_camera():
    global stop_flag
    cv2.namedWindow("Front Camera", cv2.WINDOW_AUTOSIZE)
    while not stop_flag:
        if camera_image is not None:
            cv2.imshow("Front Camera", camera_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        time.sleep(0.05)
    cv2.destroyAllWindows()

def main():
    global stop_flag

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Lidar
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('points_per_second', '56000')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('range', '50')
    lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)
    lidar_sensor.listen(lidar_callback)

    # Camera depan
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    camera_sensor.listen(camera_callback)

    # Thread visualisasi
    lidar_thread = threading.Thread(target=visualize_lidar)
    camera_thread = threading.Thread(target=visualize_camera)
    lidar_thread.start()
    camera_thread.start()

    try:
        print("Tekan Ctrl+C untuk mengakhiri.")
        while lidar_thread.is_alive() and camera_thread.is_alive():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Menghentikan visualisasi...")
        stop_flag = True

    lidar_sensor.stop()
    camera_sensor.stop()
    lidar_sensor.destroy()
    camera_sensor.destroy()
    vehicle.destroy()

    lidar_thread.join()
    camera_thread.join()
    print("Selesai.")

if __name__ == "__main__":
    main()