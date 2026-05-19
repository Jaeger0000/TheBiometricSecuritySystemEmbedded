import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import serial
import os
import glob
import urllib.request

# ==========================================
# --- AYARLAR (Raspberry Pi Versiyonu) ---
# ==========================================

# Arduino ile haberleşmeyi aç/kapat
ENABLE_SERIAL = True

# Headless calisma: Bilgisayar ekrani kapali, tum durumlar Arduino ekranina gonderilir
SHOW_DISPLAY = False

# Performans için kamera çözünürlüğü (Pi'da düşürmek FPS'i artırır)
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Seri port ayarları
# Pi'da Arduino USB ile bağlıysa: /dev/ttyACM0 (Uno R4, Mega vb.)
# USB-Serial adaptör ise: /dev/ttyUSB0
# GPIO UART (TX=GPIO14, RX=GPIO15) ise: /dev/ttyAMA0 veya /dev/serial0
SERIAL_PORT_CANDIDATES = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB0', '/dev/ttyUSB1']
BAUD_RATE = 115200

# ==========================================
# --- SERİ PORT OTOMATİK BULMA ---
# ==========================================
def auto_detect_serial_port():
    """Pi'da bağlı olan Arduino portunu otomatik bulur."""
    for port in SERIAL_PORT_CANDIDATES:
        if os.path.exists(port):
            return port
    # Yedek tarama
    found = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*')
    return found[0] if found else None

# ==========================================
# --- MODEL İNDİRİCİ ---
# ==========================================
# Modelleri script ile aynı klasöre koymak Pi'da en güvenli yöntem
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL_PATH = os.path.join(SCRIPT_DIR, 'blaze_face_short_range.tflite')
HAND_MODEL_PATH = os.path.join(SCRIPT_DIR, 'hand_landmarker.task')

def download_model(url, filename):
    if not os.path.exists(filename):
        print(f"[{os.path.basename(filename)}] bulunamadi. Indiriliyor...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"[{os.path.basename(filename)}] basariyla indirildi!")
        except Exception as e:
            print(f"HATA: Model indirilemedi - {e}")
            print("Internete baglanin veya modelleri manuel olarak ekleyin.")
            exit(1)

download_model(
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    FACE_MODEL_PATH
)
download_model(
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    HAND_MODEL_PATH
)

# ==========================================
# --- SERİ HABERLEŞME KURULUMU ---
# ==========================================
arduino = None
if ENABLE_SERIAL:
    port = auto_detect_serial_port()
    if port is None:
        print("UYARI: Arduino bulunamadi. Seri haberlesme devre disi.")
        ENABLE_SERIAL = False
    else:
        try:
            arduino = serial.Serial(port, BAUD_RATE, timeout=1)
            time.sleep(2)  # Arduino reset olur, beklemek sart
            print(f"Arduino baglandi: {port}")
        except Exception as e:
            print(f"Seri port hatasi ({port}): {e}")
            ENABLE_SERIAL = False

def send_serial(command):
    """Arduino'ya guvenli komut gonderir. Baglanti kopuksa arduino=None yapar."""
    global arduino
    if ENABLE_SERIAL and arduino is not None:
        try:
            arduino.write(command)
        except Exception as e:
            print(f"Seri yazma hatasi: {e} — Arduino baglantisi koptu.")
            try:
                arduino.close()
            except Exception:
                pass
            arduino = None


arduino_last_reconnect = 0.0
ARDUINO_RECONNECT_INTERVAL = 5.0


def try_reconnect_arduino():
    """Arduino yoksa periyodik olarak yeniden baglanmaya calisir."""
    global arduino, arduino_last_reconnect, last_serial_status
    if not ENABLE_SERIAL or arduino is not None:
        return
    now = time.time()
    if now - arduino_last_reconnect < ARDUINO_RECONNECT_INTERVAL:
        return
    arduino_last_reconnect = now
    port = auto_detect_serial_port()
    if port is None:
        return
    try:
        new_conn = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)
        arduino = new_conn
        last_serial_status = ""  # mevcut durumu tekrar gonder
        print(f"Arduino yeniden baglandi: {port}")
    except Exception as e:
        print(f"Arduino yeniden baglama denemesi basarisiz: {e}")


last_serial_status = ""


def _sanitize_for_serial(text):
    return str(text).replace("|", "/").replace("\n", " ").strip()


def send_status(line1, line2="", line3="", line4=""):
    """Arduino ekrani icin 4 satirlik durum paketi yollar."""
    global last_serial_status
    payload = "STATUS|{}|{}|{}|{}\n".format(
        _sanitize_for_serial(line1),
        _sanitize_for_serial(line2),
        _sanitize_for_serial(line3),
        _sanitize_for_serial(line4),
    )
    if payload == last_serial_status:
        return
    send_serial(payload.encode("utf-8"))
    last_serial_status = payload

# ==========================================
# --- MEDIAPIPE TASKS API KURULUMU ---
# ==========================================
face_base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
face_options = vision.FaceDetectorOptions(base_options=face_base_options)
face_detector = vision.FaceDetector.create_from_options(face_options)

hand_base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
hand_options = vision.HandLandmarkerOptions(base_options=hand_base_options, num_hands=1)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

# ==========================================
# --- SİSTEM DURUMLARI (STATE MACHINE) ---
# ==========================================
STATE_IDLE = 0          # Kilitli, yuz yok
STATE_AUTH = 1          # Yuz var, kilitli, sifre veya admin komutu bekliyor
STATE_UNLOCKED = 2      # Kilit acik
STATE_SETTING_PASS = 3  # Admin modu: Yeni 4'lu sifre bekleniyor

current_state = STATE_IDLE

# Sifre Degiskenleri
TARGET_SEQUENCE = ["Fist", "Peace", "Open", "Fist"]
ADMIN_SEQUENCE = ["Fist", "Open", "Fist"]
current_sequence = []
new_password_buffer = []

# Zamanlama ve Debounce
last_gesture_time = 0
sequence_timeout = 5.0
gesture_cooldown = 2.5
REQUIRED_CONSECUTIVE_FRAMES = 10
current_gesture_frames = 0
candidate_gesture = None

# Performans: Her N karede bir inference yap, aradaki karelerde son sonucu kullan
TARGET_FPS = 15
INFERENCE_EVERY_N_FRAMES = 2

# ==========================================
# --- JEST TANIMA FONKSIYONU ---
# ==========================================
def get_gesture(landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Basparmak
    if landmarks[tip_ids[0]].x > landmarks[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Diger parmaklar
    for i in range(1, 5):
        if landmarks[tip_ids[i]].y < landmarks[tip_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # Jestleri siniflandir
    if fingers == [0, 0, 0, 0, 0] or fingers == [1, 0, 0, 0, 0]:
        return "Fist"
    elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
        return "Peace"
    elif fingers == [1, 1, 1, 1, 1] or fingers[1:] == [1, 1, 1, 1]:
        return "Open"

    return "Unknown"

# ==========================================
# --- KAMERA KURULUMU ---
# ==========================================
CAMERA_MAX_FAILURES = 10
CAMERA_RECONNECT_DELAY = 2.0
camera_fail_count = 0


def open_camera():
    """Kamerayı acip dondurur. Basarisizsa None dondurur."""
    c = cv2.VideoCapture(0)
    c.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return c if c.isOpened() else None


cap = open_camera()
if cap is None:
    print("HATA: Kamera acilamadi. USB kameranin Pi'a bagli oldugundan emin olun.")
    exit(1)

print("=" * 50)
print("Air-Pass Security System - Raspberry Pi")
print("=" * 50)
print(f"Display: {'AKTIF' if SHOW_DISPLAY else 'KAPALI (Headless)'}")
print(f"Serial:  {'AKTIF' if ENABLE_SERIAL else 'KAPALI'}")
print("Cikis icin: 'q' tusu (display acikken) veya Ctrl+C")
print("=" * 50)

# ==========================================
# --- ANA DONGU ---
# ==========================================
try:
    frame_counter = INFERENCE_EVERY_N_FRAMES - 1
    last_face_result = None
    last_hand_result = None
    _frame_start = time.time()
    send_status("SYSTEM ACTIVE", "Waiting for face", "", "")

    while True:
        # Arduino yoksa yeniden baglanmayı dene
        try_reconnect_arduino()

        frame_counter += 1
        run_inference = (frame_counter % INFERENCE_EVERY_N_FRAMES == 0)

        success, img = cap.read()
        if not success:
            camera_fail_count += 1
            if camera_fail_count >= CAMERA_MAX_FAILURES:
                print("Kamera baglantisi koptu! Yeniden baglanmaya calisiliyor...")
                # Guvenlik: kamera yokken kilitle
                if current_state != STATE_IDLE:
                    if current_state == STATE_UNLOCKED:
                        send_serial(b'LOCK\n')
                        send_status("CAMERA ERROR", "System locked", "", "")
                    current_state = STATE_IDLE
                    current_sequence = []
                    new_password_buffer = []
                    current_gesture_frames = 0
                cap.release()
                time.sleep(CAMERA_RECONNECT_DELAY)
                new_cap = open_camera()
                if new_cap is not None:
                    cap = new_cap
                    camera_fail_count = 0
                    print("Kamera yeniden baglandi.")
                # camera_fail_count >= CAMERA_MAX_FAILURES kalirsa bir sonraki turda tekrar dener
            else:
                time.sleep(0.1)
            continue
        camera_fail_count = 0

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        h, w, c = img.shape

        # ==========================================
        # 1. YUZ ALGILAMA VE OTOMATIK KILITLEME
        # ==========================================
        if run_inference:
            face_result = face_detector.detect(mp_image)
            last_face_result = face_result
        else:
            face_result = last_face_result

        if len(face_result.detections) > 0:
            if current_state == STATE_IDLE:
                current_state = STATE_AUTH
                print("Yuz Algilandi. Sistem Aktif.")
                send_status("FACE DETECTED", "Enter gesture...", str(TARGET_SEQUENCE), "")
        else:
            if current_state != STATE_IDLE:
                was_unlocked = (current_state == STATE_UNLOCKED)
                print("Yuz kayboldu! Sistem otomatik kilitlendi.")
                current_state = STATE_IDLE
                current_sequence = []
                new_password_buffer = []
                current_gesture_frames = 0
                send_status("SYSTEM LOCKED", "Waiting for face", "", "")
                if was_unlocked:
                    send_serial(b'LOCK\n')

        # ==========================================
        # 2. DURUM MAKINESI
        # ==========================================
        if current_state == STATE_IDLE:
            pass

        elif current_state == STATE_AUTH or current_state == STATE_SETTING_PASS:

            # El algilama
            if run_inference:
                hand_result = hand_landmarker.detect(mp_image)
                last_hand_result = hand_result
            else:
                hand_result = last_hand_result
            if len(hand_result.hand_landmarks) > 0:
                landmarks = hand_result.hand_landmarks[0]

                detected_gesture = get_gesture(landmarks)

                # Debounce
                if detected_gesture != "Unknown":
                    if detected_gesture == candidate_gesture:
                        current_gesture_frames += 1
                    else:
                        candidate_gesture = detected_gesture
                        current_gesture_frames = 1

                    if current_gesture_frames >= REQUIRED_CONSECUTIVE_FRAMES:
                        current_time = time.time()

                        if (current_time - last_gesture_time) > gesture_cooldown:

                            # MOD 1: Sifre veya admin komutu
                            if current_state == STATE_AUTH:
                                temp_seq = current_sequence + [candidate_gesture]
                                is_admin_path = (ADMIN_SEQUENCE[:len(temp_seq)] == temp_seq)
                                is_target_path = (TARGET_SEQUENCE[:len(temp_seq)] == temp_seq)

                                if not (is_admin_path or is_target_path):
                                    print(f"Wrong Move ({candidate_gesture}). Sequence reset.")
                                    send_status(
                                        f"Wrong Move ({candidate_gesture}). Sequence reset.",
                                        "",
                                        "",
                                        "",
                                    )
                                    current_sequence = []
                                else:
                                    current_sequence.append(candidate_gesture)
                                    print(f"Step Successful: {candidate_gesture}. Status: {current_sequence}")
                                    send_status(
                                        f"Step OK: {candidate_gesture}",
                                        f"Progress: {len(current_sequence)}/{len(TARGET_SEQUENCE)}",
                                        "WAIT...",
                                        "",
                                    )

                                    if current_sequence == ADMIN_SEQUENCE:
                                        print("--- ADMIN MODE ACTIVE --- Enter the new 4-step passcode.")
                                        current_state = STATE_SETTING_PASS
                                        current_sequence = []
                                        new_password_buffer = []

                                    elif current_sequence == TARGET_SEQUENCE:
                                        print("ACCESS GRANTED! Lock opened.")
                                        current_state = STATE_UNLOCKED
                                        current_sequence = []
                                        send_status("ACCESS GRANTED!", "Lock opened", "", "")
                                        send_serial(b'UNLOCK\n')

                            # MOD 2: Yeni sifreyi belirleme
                            elif current_state == STATE_SETTING_PASS:
                                new_password_buffer.append(candidate_gesture)
                                print(f"New Passcode Step: {candidate_gesture} ({len(new_password_buffer)}/4)")
                                send_status(
                                    f"Step OK: {candidate_gesture}",
                                    f"Progress: {len(new_password_buffer)}/4",
                                    "WAIT...",
                                    "",
                                )

                                if len(new_password_buffer) == 4:
                                    TARGET_SEQUENCE = new_password_buffer.copy()
                                    print(f"SUCCESS! New passcode saved: {TARGET_SEQUENCE}")
                                    current_state = STATE_AUTH
                                    new_password_buffer = []

                            last_gesture_time = current_time
                            time.sleep(gesture_cooldown)
                            if current_state == STATE_AUTH and len(current_sequence) < len(TARGET_SEQUENCE):
                                done_str = " > ".join(current_sequence)
                                send_status(
                                    f"STEP {len(current_sequence)+1}/{len(TARGET_SEQUENCE)}",
                                    done_str,
                                    "",
                                    "",
                                )
                            elif current_state == STATE_SETTING_PASS and len(new_password_buffer) < 4:
                                done_str = " > ".join(new_password_buffer)
                                send_status(
                                    f"STEP {len(new_password_buffer)+1}/4",
                                    done_str,
                                    "",
                                    "",
                                )

                        current_gesture_frames = 0

            # Guvenlik zaman asimi
            if (current_state == STATE_AUTH and len(current_sequence) > 0) or \
                    (current_state == STATE_SETTING_PASS and len(new_password_buffer) > 0):
                if (time.time() - last_gesture_time) > sequence_timeout:
                    print("Timeout! Operation canceled.")
                    current_sequence = []
                    new_password_buffer = []
                    current_state = STATE_AUTH

        elif current_state == STATE_UNLOCKED:
            send_status("DOOR OPEN", "Detecting face...", "", "")

        # FPS sinirla ve gercek FPS'i terminale yaz
        elapsed = time.time() - _frame_start
        sleep_time = (1.0 / TARGET_FPS) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        actual_fps = 1.0 / max(time.time() - _frame_start, 0.001)
        print(f"FPS: {actual_fps:.1f}", end="\r")
        _frame_start = time.time()

except KeyboardInterrupt:
    print("\nKullanici tarafindan durduruldu (Ctrl+C).")

finally:
    # ==========================================
    # --- TEMIZLIK ---
    # ==========================================
    print("Sistem kapatiliyor...")
    cap.release()
    if SHOW_DISPLAY:
        cv2.destroyAllWindows()
    if ENABLE_SERIAL and arduino is not None:
        send_serial(b'LOCK\n')  # Cikistaki kilitle
        arduino.close()
    print("Temizlik tamamlandi.")