import carla
import open3d as o3d
import numpy as np
import cv2
import threading
import time
import random

# Variabel global untuk sensor
latest_image = None
latest_points = None
stop_threads = False

def camera_callback(image):
    global latest_image
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((image.height, image.width, 4))
    img = img[:, :, :3]  # Ambil RGB, buang alpha
    latest_image = img

def lidar_callback(point_cloud):
    global latest_points
    points = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)[:, :3]
    latest_points = points
    print("Jumlah titik lidar:", points.shape[0])

def camera_window():
    global stop_threads
    while not stop_threads:
        if latest_image is not None:
            cv2.imshow("Kamera Depan Mobil", latest_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_threads = True
                break
        else:
            time.sleep(0.05)
    cv2.destroyAllWindows()

def lidar_window():
    global stop_threads
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Lidar Point Cloud")
    render_option = vis.get_render_option()
    render_option.point_size = 5.0  # Titik diperbesar
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    while not stop_threads:
        if latest_points is not None and len(latest_points) > 0:
            pcd.points = o3d.utility.Vector3dVector(latest_points)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.03)
        else:
            time.sleep(0.05)
    vis.destroy_window()

def main():
    global stop_threads
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle')[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    # Setup kamera RGB
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Setup lidar
    lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
    lidar_bp.set_attribute('channels', '32')
    lidar_bp.set_attribute('points_per_second', '56000')
    lidar_bp.set_attribute('rotation_frequency', '10')
    lidar_bp.set_attribute('range', '50')
    lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
    lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

    # Listen ke sensor
    camera.listen(camera_callback)
    lidar_sensor.listen(lidar_callback)

    # Mulai 2 thread window
    t1 = threading.Thread(target=camera_window)
    t2 = threading.Thread(target=lidar_window)
    t1.start()
    t2.start()

    try:
        print("Tekan Ctrl+C di terminal untuk berhenti, atau 'q' di window kamera.")
        while not stop_threads:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
        stop_threads = True
    finally:
        stop_threads = True
        t1.join()
        t2.join()
        camera.stop()
        lidar_sensor.stop()
        camera.destroy()
        lidar_sensor.destroy()
        vehicle.destroy()

if __name__ == "__main__":
    main()