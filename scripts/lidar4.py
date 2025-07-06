import carla
import open3d as o3d
import numpy as np
import threading
import cv2
import time

latest_image = None
latest_points = None
stop_flag = False

def camera_callback(image):
    global latest_image
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    rgb = array[:, :, :3][:, :, ::-1]
    latest_image = rgb

def lidar_callback(point_cloud):
    global latest_points
    array = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
    array = np.reshape(array, (-1, 4))
    latest_points = array[:, :3]

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

def show_lidar_live():
    global latest_points, stop_flag
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Lidar Point Cloud", width=700, height=700)
    pcd = o3d.geometry.PointCloud()
    added = False
    while not stop_flag:
        if latest_points is not None:
            pcd.points = o3d.utility.Vector3dVector(latest_points)
            if not added:
                vis.add_geometry(pcd)
                added = True
            else:
                vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
        time.sleep(0.03)
    vis.destroy_window()

def main():
    global stop_flag
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle')[0]
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Lidar setup: FOV 87° depan, tidak berputar
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('points_per_second', '20000')
    lidar_bp.set_attribute('rotation_frequency', '2000')    # Tidak berputar
    lidar_bp.set_attribute('range', '50')
    lidar_bp.set_attribute('horizontal_fov', '87')
    lidar_bp.set_attribute('upper_fov', '10')
    lidar_bp.set_attribute('lower_fov', '-10')
    lidar_location = carla.Location(x=0, z=2.5)
    lidar_transform = carla.Transform(lidar_location)
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

    # Kamera RGB
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '640')
    camera_bp.set_attribute('image_size_y', '480')
    camera_bp.set_attribute('fov', '90')
    camera_location = carla.Location(x=1.5, z=2.4)
    camera_transform = carla.Transform(camera_location)
    camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Start sensor
    camera_sensor.listen(camera_callback)
    lidar_sensor.listen(lidar_callback)

    # Start threads untuk masing-masing window
    cam_thread = threading.Thread(target=show_camera_thread, daemon=True)
    lidar_thread = threading.Thread(target=show_lidar_live, daemon=True)
    cam_thread.start()
    lidar_thread.start()

    print("Tekan Q pada window kamera untuk keluar (kedua window akan tertutup).")
    # Tunggu sampai user menutup window kamera
    while cam_thread.is_alive() and lidar_thread.is_alive() and not stop_flag:
        time.sleep(0.1)

    # Cleanup
    lidar_sensor.stop()
    lidar_sensor.destroy()
    camera_sensor.stop()
    camera_sensor.destroy()
    vehicle.destroy()

if __name__ == "__main__":
    main()