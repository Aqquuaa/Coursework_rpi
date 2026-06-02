from flask import Flask, Response
import cv2
import threading
import time
import math
import os
import numpy as np
from pymavlink import mavutil

app = Flask(__name__)

OUT_OF_BOUNDS = -9999

# --- Налаштування DEM ---
DEM_FILE = 'N50E030.hgt'
SAMPLES = 3601
TILE_LAT = 50.0
TILE_LON = 30.0

# Завантажуємо цифрову карту висот у пам'ять
# Якщо файл відсутній, то створюємо плаский масив на висоті 150 метрів, щоб програма працювала
if os.path.exists(DEM_FILE):
    print(f"[DEM] Loading {DEM_FILE} into memory...")
    with open(DEM_FILE, 'rb') as f:
        # >i2 = Big-endian 16-bit integer (стандартний SRTM формат)
        dem_grid = np.fromfile(f, np.dtype('>i2'), SAMPLES * SAMPLES).reshape((SAMPLES, SAMPLES))
else:
    print(f"[DEM WARNING] {DEM_FILE} not found. Generating flat 150m dummy terrain.")
    dem_grid = np.full((SAMPLES, SAMPLES), 150, dtype=np.int16)

# --- Функція пошуку висоти ---
# дає висоту у точці за О(1) час у метрах
def get_elevation(lat, lon):
    # для пошуку висоти за О(1) час спочатку знаходимо ділянку саме на карті висот
    lat_offset = lat - TILE_LAT
    lon_offset = lon - TILE_LON

    # для безпеки перевіряємо чи шукана точка не знаходиться за межами карти
    if not (0 <= lat_offset <= 1.0 and 0 <= lon_offset <= 1.0):
        return OUT_OF_BOUNDS

    # Оскільки рядки карти йдуть з півночі на південь, інвертуємо вісь У
    row = int((1.0 - lat_offset) * (SAMPLES - 1))
    col = int(lon_offset * (SAMPLES - 1))

    # обмеження для запобігання помилці виходу за масив
    row = max(0, min(SAMPLES - 1, row))
    col = max(0, min(SAMPLES - 1, col))

    return dem_grid[row, col]

# --- Загальний стан ---
telemetry_data = {
    'lat': 0.0,
    'lon': 0.0,
    'alt_amsl': 0.0, # абсолютна висота
    'pitch': 0.0,
    'yaw': 0.0,
    'connected': False
}

# --- Потік обробки MAVLink ---
def mavlink_thread():
    print("[MAVLINK] Waiting for flight controller on /dev/ttyACM0...")
    try:
        master = mavutil.mavlink_connection('/dev/ttyACM0', baud=115200)
        master.wait_heartbeat()
        print("[MAVLINK] Heartbeat detected!")
        telemetry_data['connected'] = True

        # При наявності зв'язку з польотним контролером постійно збираємо телемерію
        while True:
            msg = master.recv_match(type=['GLOBAL_POSITION_INT', 'ATTITUDE'], blocking=True, timeout=1.0)
            if msg:
                msg_type = msg.get_type()
                if msg_type == 'GLOBAL_POSITION_INT':
                    telemetry_data['lat'] = msg.lat / 1e7
                    telemetry_data['lon'] = msg.lon / 1e7
                    # використовується абсолютна висота у мм, а не відносна
                    telemetry_data['alt_amsl'] = msg.alt / 1000.0
                elif msg_type == 'ATTITUDE':
                    telemetry_data['pitch'] = msg.pitch
                    telemetry_data['yaw'] = msg.yaw
            time.sleep(0.005)

    except Exception as e:
        print(f"[MAVLINK ERROR] {e}")
        telemetry_data['connected'] = False

# --- Функція генерації кадрів ---
def generate_frames():
    cap = cv2.VideoCapture(0)
    R_E = 6378137.0 # Радіус землі у м для розрахунків

    # Константи трасування променів
    MAX_RANGE = 5000.0   # Обмеження у 5km
    COARSE_STEP = 25.0   # довжина кроку для першого етапу

    while True:
        success, frame = cap.read()
        if not success:
            break

        # масштабування кадру для зменшення навантаження
        frame = cv2.resize(frame, (960, 540))
        h, w, _ = frame.shape
        center_x, center_y = int(w/2), int(h/2)

        # Прицільна сітка по центру екрану
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 40, (0, 255, 0), 1)

        # Якщо немає з'єднання з польотним контролером, не робимо розрахунки
        if telemetry_data['connected']:
            uav_lat = telemetry_data['lat']
            uav_lon = telemetry_data['lon']
            uav_alt = telemetry_data['alt_amsl']
            pitch = telemetry_data['pitch']
            yaw = telemetry_data['yaw']

            # Вектор камери
            v_N = math.cos(pitch) * math.cos(yaw)
            v_E = math.cos(pitch) * math.sin(yaw)
            v_D = -math.sin(pitch)

            # --- Трасування променів ---
            hit = False
            current_dist = 0.0
            tgt_lat, tgt_lon, tgt_alt = 0.0, 0.0, 0.0

            # Йде обрахунок лише якщо камера дивиться вниз
            if v_D > -0.1:
                lat_rad = math.radians(uav_lat)
                cos_lat = math.cos(lat_rad)

                # Перша частина: крокування з меншою точністю
                while current_dist <= MAX_RANGE:
                    #Розрахунок географічного зсуву математичного променя
                    delta_lat = ((current_dist * v_N) / R_E) * (180.0 / math.pi)
                    delta_lon = ((current_dist * v_E) / (R_E * cos_lat)) * (180.0 / math.pi)

                    test_lat = uav_lat + delta_lat
                    test_lon = uav_lon + delta_lon
                    ray_alt = uav_alt - (current_dist * v_D)

                    ground_alt = get_elevation(test_lat, test_lon)

                    # якщо точка за межами карти зупиняємося
                    if ground_alt == OUT_OF_BOUNDS:
                        break

                    # якщо промінь нижче землі, значить було "влучання"
                    if ray_alt <= ground_alt:
                        hit = True
                        break

                    current_dist += COARSE_STEP

                # Друга частина: якщо було влучання, шукаємо точну позицію
                if hit:
                    low_dist = current_dist - COARSE_STEP
                    high_dist = current_dist

                    # 5 ітерацій бінарного пошуку, які дадуть точність менше метра
                    for _ in range(5):
                        mid_dist = (low_dist + high_dist) / 2.0
                        delta_lat = ((mid_dist * v_N) / R_E) * (180.0 / math.pi)
                        delta_lon = ((mid_dist * v_E) / (R_E * cos_lat)) * (180.0 / math.pi)

                        test_lat = uav_lat + delta_lat
                        test_lon = uav_lon + delta_lon
                        ray_alt = uav_alt - (mid_dist * v_D)
                        ground_alt = get_elevation(test_lat, test_lon)

                        if ray_alt <= ground_alt:
                            high_dist = mid_dist # все ще нижче землі, повертаємося
                        else:
                            low_dist = mid_dist  # вище землі, йдемо далі

                    # фінальні координати
                    current_dist = mid_dist
                    tgt_lat = test_lat
                    tgt_lon = test_lon

            # --- Оновлення наекранного інтерфейсу ---
            if hit:
                tgt_text_lat = f"LAT:  {tgt_lat:.6f}"
                tgt_text_lon = f"LON:  {tgt_lon:.6f}"
                tgt_text_dist = f"DIST: {current_dist:.1f}m"
                tgt_color = (0, 255, 255)
            else:
                tgt_text_lat = "LAT:  HORIZON/OOR"
                tgt_text_lon = "LON:  HORIZON/OOR"
                tgt_text_dist = "DIST: >5000m"
                tgt_color = (0, 0, 255)

            uav_text_lat = f"LAT: {uav_lat:.6f}"
            uav_text_lon = f"LON: {uav_lon:.6f}"
            uav_text_alt = f"AMSL: {uav_alt:.1f}m"
            uav_color = (0, 255, 0)
        else:
            uav_text_lat = uav_text_lon = uav_text_alt = "NO DATA"
            tgt_text_lat = tgt_text_lon = tgt_text_dist = "NO DATA"
            uav_color = tgt_color = (0, 0, 255)

        # Вимальовка тексту телеметрії
        cv2.putText(frame, "UAV POS:", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, uav_color, 2)
        cv2.putText(frame, uav_text_lat, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, uav_color, 2)
        cv2.putText(frame, uav_text_lon, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, uav_color, 2)
        cv2.putText(frame, uav_text_alt, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, uav_color, 2)

        cv2.putText(frame, "TARGET POS:", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tgt_color, 2)
        cv2.putText(frame, tgt_text_lat, (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tgt_color, 2)
        cv2.putText(frame, tgt_text_lon, (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tgt_color, 2)
        cv2.putText(frame, tgt_text_dist, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tgt_color, 2)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    t = threading.Thread(target=mavlink_thread, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False)