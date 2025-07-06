import carla
import open3d as o3d
import numpy as np
import cv2
import threading
import time

latest_image = None
stop_flag = False
cloud_ready = False
points_data = []

def camera_callback(image):
    global latest_image
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    rgb = array[:, :, :3][:, :, ::-1]  # Convert to RGB
    latest_image = rgb

def show_camera_thread():
    global latest_image, stop_flag
    while not stop_flag:
        if latest_image is not None:
            cv2.imshow("Camera View", latest_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_flag = True
                break
        time.sleep(0.01)
    cv2.destroyAllWindows()

def lidar_callback(point_cloud):
    global points_data, cloud_ready
    array = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
    array = np.reshape(array, (-1, 4))
    if not cloud_ready:
        points_data = array[:, :3]
        cloud_ready = True

def main():
    global stop_flag, cloud_ready, points_data
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Setup Lidar: FOV 87 derajat ke depan, tidak berputar
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('points_per_second', '20000')
    lidar_bp.set_attribute('rotation_frequency', '0')    # Tidak berputar
    lidar_bp.set_attribute('range', '50')
    lidar_bp.set_attribute('horizontal_fov', '87')       # Hanya 87 derajat ke depan
    lidar_bp.set_attribute('upper_fov', '10')            # Vertical FOV: -10 s/d +10 derajat
    lidar_bp.set_attribute('lower_fov', '-10')
    lidar_location = carla.Location(x=0, z=2.5)
    lidar_transform = carla.Transform(lidar_location)
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

    # Setup Kamera RGB di atas mobil
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '480')
    camera_bp.set_attribute('fov', '90')
    camera_location = carla.Location(x=1.5, z=2.4)
    camera_transform = carla.Transform(camera_location)
    camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Start Kamera
    camera_sensor.listen(camera_callback)
    cam_thread = threading.Thread(target=show_camera_thread, daemon=True)
    cam_thread.start()

    # Start Lidar
    lidar_sensor.listen(lidar_callback)

    print("Tunggu 2 detik untuk mendapatkan satu frame Lidar dan stream kamera...")
    waited = 0
    while not cloud_ready and waited < 5:
        time.sleep(0.1)
        waited += 0.1

    # Visualisasi Lidar (Open3D)
    if cloud_ready:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_data)
        o3d.visualization.draw_geometries([pcd], window_name="Lidar Point Cloud FOV 87 deg")

    print("Tekan Q pada window kamera untuk keluar.")
    # Tunggu thread kamera selesai
    while cam_thread.is_alive() and not stop_flag:
        time.sleep(0.1)

    # Cleanup
    lidar_sensor.stop()
    lidar_sensor.destroy()
    camera_sensor.stop()
    camera_sensor.destroy()
    vehicle.destroy()

if __name__ == "__main__":
    main()