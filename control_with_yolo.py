import time
import airsim
import keyboard
import numpy as np
import cv2
from ultralytics import YOLO

vehicle_name = "SimpleFlight"

# -----------------------------
# Настройки управления
# -----------------------------
VEL = 100.0          # скорость по X/Y, м/с
YAW_RATE = 45.0    # скорость поворота, град/с
Z_TARGET = -30.0    # рабочая высота
LOOP_DT = 0.08     # шаг цикла

# -----------------------------
# Настройки YOLO
# -----------------------------
YOLO_MODEL_PATH = "spark_2.pt"   # путь к вашей модели
CAMERA_NAME = "2"             # имя камеры AirSim
YOLO_IMGSZ = 640
YOLO_CONF = 0.25
YOLO_IOU = 0.6

# None = не фильтровать, выводить все классы модели
TARGET_CLASS_NAMES = None

# Запускать детекцию на каждом N-м кадре
DETECTION_EVERY_N_FRAMES = 1

client = airsim.MultirotorClient()
client.confirmConnection()

print("Загрузка YOLO...")
model = YOLO(YOLO_MODEL_PATH)
print("YOLO classes:", model.names)

client.enableApiControl(True, vehicle_name)
client.armDisarm(True, vehicle_name)
client.takeoffAsync(vehicle_name=vehicle_name).join()
client.moveToZAsync(Z_TARGET, 1.5, vehicle_name=vehicle_name).join()

print("Управление:")
print("W/S - вперед/назад")
print("A/D - влево/вправо")
print("Q/E - yaw")
print("R/F - выше/ниже")
print("ESC - выход")

last_print_time = 0.0
prev_collision = False
frame_idx = 0
last_detections = []


def get_scene_image_bgr():
    responses = client.simGetImages(
        [
            airsim.ImageRequest(
                CAMERA_NAME,
                airsim.ImageType.Scene,
                pixels_as_float=False,
                compress=False,
            )
        ],
        vehicle_name=vehicle_name
    )

    if not responses:
        return None

    resp = responses[0]
    if resp.width <= 0 or resp.height <= 0 or len(resp.image_data_uint8) == 0:
        return None

    img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
    if img.size != resp.height * resp.width * 3:
        return None

    img = img.reshape(resp.height, resp.width, 3)
    return img


def run_yolo(image_bgr):
    results = model.predict(
        source=image_bgr,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        verbose=False,
    )

    detections = []
    if not results:
        return detections

    result = results[0]
    boxes = result.boxes
    names = result.names if hasattr(result, "names") else {}

    if boxes is None:
        return detections

    xyxy = boxes.xyxy.detach().cpu().numpy() if boxes.xyxy is not None else []
    confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else []
    clss = boxes.cls.detach().cpu().numpy() if boxes.cls is not None else []

    h, w = image_bgr.shape[:2]

    for box, conf, cls_id in zip(xyxy, confs, clss):
        x1, y1, x2, y2 = [int(v) for v in box]
        cls_id = int(cls_id)
        class_name = str(names.get(cls_id, cls_id)).lower()

        if TARGET_CLASS_NAMES is not None and class_name not in TARGET_CLASS_NAMES:
            continue

        bw = max(0, x2 - x1)
        bh = max(0, y2 - y1)
        area = bw * bh
        area_norm = area / float(max(w * h, 1))

        detections.append({
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "conf": float(conf),
            "class_id": cls_id,
            "class_name": class_name,
            "center_x": float((x1 + x2) / 2.0),
            "center_y": float((y1 + y2) / 2.0),
            "area": int(area),
            "area_norm": float(area_norm),
        })

    return detections


def draw_detections(frame, detections):
    out = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        label = f'{det["class_name"]} {det["conf"]:.2f}'

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cx, cy = int(det["center_x"]), int(det["center_y"])
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

    return out


try:
    current_z = Z_TARGET

    while True:
        if keyboard.is_pressed("esc"):
            break

        vx = 0.0
        vy = 0.0
        yaw_rate = 0.0

        # -----------------------------
        # Управление по X/Y
        # -----------------------------
        if keyboard.is_pressed("w"):
            vx += VEL
        if keyboard.is_pressed("s"):
            vx -= VEL
        if keyboard.is_pressed("d"):
            vy += VEL
        if keyboard.is_pressed("a"):
            vy -= VEL

        # -----------------------------
        # Поворот
        # -----------------------------
        if keyboard.is_pressed("e"):
            yaw_rate += YAW_RATE
        if keyboard.is_pressed("q"):
            yaw_rate -= YAW_RATE

        # -----------------------------
        # Высота
        # -----------------------------
        if keyboard.is_pressed("r"):
            current_z -= 0.05
        if keyboard.is_pressed("f"):
            current_z += 0.05

        # -----------------------------
        # Команда дрону
        # -----------------------------
        client.moveByVelocityZAsync(
            vx,
            vy,
            current_z,
            LOOP_DT,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
            vehicle_name=vehicle_name
        )

        time.sleep(LOOP_DT)

        # -----------------------------
        # Чтение collision
        # -----------------------------
        collision = client.simGetCollisionInfo(vehicle_name=vehicle_name)

        # -----------------------------
        # Чтение датчиков
        # -----------------------------
        front = client.getDistanceSensorData("FrontDistance", vehicle_name)
        left = client.getDistanceSensorData("LeftDistance", vehicle_name)
        right = client.getDistanceSensorData("RightDistance", vehicle_name)
        up = client.getDistanceSensorData("UpDistance", vehicle_name)
        down = client.getDistanceSensorData("DownDistance", vehicle_name)

        # -----------------------------
        # Чтение позиции
        # -----------------------------
        state = client.getMultirotorState(vehicle_name=vehicle_name)
        pos = state.kinematics_estimated.position

        # -----------------------------
        # Видео + YOLO
        # -----------------------------
        frame = get_scene_image_bgr()

        if frame is not None:
            if frame_idx % DETECTION_EVERY_N_FRAMES == 0:
                last_detections = run_yolo(frame)

                if last_detections:
                    print("\n=== DETECTIONS ===")
                    for det in last_detections:
                        print(
                            f'class={det["class_name"]} | '
                            f'class_id={det["class_id"]} | '
                            f'conf={det["conf"]:.3f} | '
                            f'bbox=({det["x1"]},{det["y1"]},{det["x2"]},{det["y2"]}) | '
                            f'center=({det["center_x"]:.1f},{det["center_y"]:.1f}) | '
                            f'area_norm={det["area_norm"]:.4f}'
                        )
                    print("==================\n")

            vis_frame = draw_detections(frame, last_detections)
            cv2.imshow("AirSim YOLO Debug", vis_frame)
            cv2.waitKey(1)

        frame_idx += 1

        # -----------------------------
        # Печать только в момент нового столкновения
        # -----------------------------
        if collision.has_collided and not prev_collision:
            print("\n=== COLLISION DETECTED ===")
            print(f"Object: {collision.object_name}")
            print(f"Drone position: X={pos.x_val:.2f}, Y={pos.y_val:.2f}, Z={pos.z_val:.2f}")
            print(
                f"Sensors at impact -> "
                f"Front={front.distance:.2f}, "
                f"Left={left.distance:.2f}, "
                f"Right={right.distance:.2f}, "
                f"Up={up.distance:.2f}, "
                f"Down={down.distance:.2f}"
            )
            print("==========================\n")

        prev_collision = collision.has_collided

        # -----------------------------
        # Периодический статус
        # -----------------------------
        now = time.time()
        if now - last_print_time > 0.5:
            print(
                f"Pos: X={pos.x_val:6.2f} Y={pos.y_val:6.2f} Z={pos.z_val:6.2f} | "
                f"Collided: {collision.has_collided} | "
                f"Obj: {collision.object_name} | "
                f"F={front.distance:4.2f} "
                f"L={left.distance:4.2f} "
                f"R={right.distance:4.2f} "
                f"U={up.distance:4.2f} "
                f"D={down.distance:4.2f}"
            )

            if last_detections:
                print(f"Detected objects: {len(last_detections)}")

            last_print_time = now

finally:
    cv2.destroyAllWindows()
    client.hoverAsync(vehicle_name=vehicle_name).join()
    client.armDisarm(False, vehicle_name)
    client.enableApiControl(False, vehicle_name)
    print("Выход")