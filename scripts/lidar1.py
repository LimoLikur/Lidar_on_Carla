import carla
import time

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# 1. Cari blueprint mobil
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle')[0]
spawn_point = world.get_map().get_spawn_points()[0]

# 2. Spawn mobil
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# 3. Pilih blueprint Lidar
lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
# (Optional) Ubah setting Lidar, misal:
lidar_bp.set_attribute('channels', '32')
lidar_bp.set_attribute('points_per_second', '56000')
lidar_bp.set_attribute('rotation_frequency', '10')
lidar_bp.set_attribute('range', '50')

# 4. Atur posisi sensor Lidar di atas mobil
lidar_location = carla.Location(x=0, z=2.5)
lidar_rotation = carla.Rotation()
lidar_transform = carla.Transform(lidar_location, lidar_rotation)

# 5. Spawn sensor Lidar
lidar_sensor = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

# 6. Callback untuk menerima data Lidar
def lidar_callback(point_cloud):
    print('Received Lidar data:', point_cloud)

lidar_sensor.listen(lambda data: lidar_callback(data))

# 7. Jalankan simulasi beberapa detik
try:
    time.sleep(10)
finally:
    lidar_sensor.stop()
    lidar_sensor.destroy()
    vehicle.destroy()