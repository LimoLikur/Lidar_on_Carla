import carla
import open3d as o3d
import numpy as np
import cv2
import threading
import time

# Ukuran window yang sama untuk kamera & lidar
WINDOW_W, WINDOW_H = 400, 300

# Variabel global agar thread bisa sharing data
latest_image = None
latest_points = None
stop_threads = False

def camera_callback(image):
    global latest_image
    # Konversi ke numpy array, reshape, resize
    img = np.frombuffer(image.raw_data, dtype=np.uint8)
    img = img.reshape((image.height, image.width, 4))
    img = img[:, :, :3]  # Buang channel alpha
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (WINDOW_W, WINDOW_H))
    latest_image = img

def lidar_callback(point_cloud):
    global latest_points
    # Ambil x, y, z, intensity
    points = np.frombuffer(point_cloud.raw_data, dtype=np.float32).reshape(-1, 4)
    latest_points = points  # (N, 4): x, y, z, intensity

def camera_window():
    global stop_threads
    cv2.namedWindow("Kamera Mobil", cv2.WINDOW_AUTOSIZE)
    while not stop_threads:
        if latest_image is not None:
            cv2.imshow("Kamera Mobil", latest_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            time.sleep(0.05)
    cv2.destroyAllWindows()

def lidar_window():
    global stop_threads
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Lidar Point Cloud", width=WINDOW_W, height=WINDOW_H, left=50, top=50)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    while not stop_threads:
        if latest_points is not None:
            xyz = latest_points[:, :3]
            intens = latest_points[:, 3]
            # Normalisasi intensitas ke 0-1
            intens_norm = (intens - intens.min()) / (intens.ptp() + 1e-6)
            # Skala ke 0-255 dan pastikan shape (N,1)
            intens_img = np.clip((intens_norm * 255), 0, 255).astype(np.uint8).reshape(-1, 1)
            # Pewarnaan colormap JET
            colors = cv2.applyColorMap(intens_img, cv2.COLORMAP_JET)
            colors = colors[:, ::-1] / 255.0  # BGR ke RGB dan ke [0,1]
            # Pastikan jumlah points = jumlah warna
            if xyz.shape[0] == colors.shape[0]:
                pcd.points = o3d.utility.Vector3dVector(xyz)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
            else:
                print("WARNING: shape mismatch! xyz:", xyz.shape, "colors:", colors.shape)
            time.sleep(0.02)
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
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    # Kamera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Lidar
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

    t1 = threading.Thread(target=camera_window)
    t2 = threading.Thread(target=lidar_window)
    t1.start()
    t2.start()

    try:
        print("Tekan Ctrl+C di terminal untuk berhenti, atau 'q' di window kamera.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
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