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

# Görüntüyü ekranda göster (HDMI/Monitor varsa True, headless/SSH ise False)
SHOW_DISPLAY = True

# Performans için kamera çözünürlüğü (Pi'da düşürmek FPS'i artırır)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Seri port ayarları
# Pi'da Arduino USB ile bağlıysa: /dev/ttyACM0 (Uno R4, Mega vb.)
# USB-Serial adaptör ise: /dev/ttyUSB0
# GPIO UART (TX=GPIO14, RX=GPIO15) ise: /dev/ttyAMA0 veya /dev/serial0
SERIAL_PORT_CANDIDATES = ['/dev/ttyACM0', '/dev/ttyACM1', '/dev/ttyUSB0', '/dev/ttyUSB1']
BAUD_RATE = 9600

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
    """Arduino'ya guvenli komut gonderir."""
    if ENABLE_SERIAL and arduino is not None:
        try:
            arduino.write(command)
        except Exception as e:
            print(f"Seri yazma hatasi: {e}")

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
gesture_cooldown = 1.5
REQUIRED_CONSECUTIVE_FRAMES = 10
current_gesture_frames = 0
candidate_gesture = None

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
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
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
    while True:
        success, img = cap.read()
        if not success:
            print("Kamera frame okunamadi, tekrar deneniyor...")
            time.sleep(0.1)
            continue

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        h, w, c = img.shape

        # ==========================================
        # 1. YUZ ALGILAMA VE OTOMATIK KILITLEME
        # ==========================================
        face_result = face_detector.detect(mp_image)

        if len(face_result.detections) > 0:
            if current_state == STATE_IDLE:
                current_state = STATE_AUTH
                print("Yuz Algilandi. Sistem Aktif.")

            bbox_color = (0, 255, 0) if current_state == STATE_UNLOCKED else (0, 0, 255)

            if SHOW_DISPLAY:
                for detection in face_result.detections:
                    bbox = detection.bounding_box
                    cv2.rectangle(img, (bbox.origin_x, bbox.origin_y),
                                  (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height),
                                  bbox_color, 2)
        else:
            if current_state != STATE_IDLE:
                print("Yuz kayboldu! Sistem otomatik kilitlendi.")
                current_state = STATE_IDLE
                current_sequence = []
                new_password_buffer = []
                current_gesture_frames = 0
                send_serial(b'LOCK\n')

        # ==========================================
        # 2. DURUM MAKINESI
        # ==========================================
        if current_state == STATE_IDLE:
            if SHOW_DISPLAY:
                cv2.putText(img, "SYSTEM LOCKED - NO FACE", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        elif current_state == STATE_AUTH or current_state == STATE_SETTING_PASS:
            if SHOW_DISPLAY:
                if current_state == STATE_AUTH:
                    cv2.putText(img, "STATE: LOCKED (Waiting for Command)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    cv2.putText(img, f"Seq: {' -> '.join(current_sequence)}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    cv2.putText(img, "STATE: ADMIN (Setting New Passcode)", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    cv2.putText(img, f"New Pass: {' -> '.join(new_password_buffer)} (Need 4)",
                                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # El algilama
            hand_result = hand_landmarker.detect(mp_image)
            if len(hand_result.hand_landmarks) > 0:
                landmarks = hand_result.hand_landmarks[0]

                if SHOW_DISPLAY:
                    for lm in landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(img, (cx, cy), 4, (0, 255, 255), cv2.FILLED)

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
                                    print(f"Yanlis Hareket ({candidate_gesture}). Dizi Sifirlandi.")
                                    current_sequence = []
                                else:
                                    current_sequence.append(candidate_gesture)
                                    print(f"Adim Basarili: {candidate_gesture}. Durum: {current_sequence}")

                                    if current_sequence == ADMIN_SEQUENCE:
                                        print("--- ADMIN MODU AKTIF --- Yeni 4'lu sifrenizi girin.")
                                        current_state = STATE_SETTING_PASS
                                        current_sequence = []
                                        new_password_buffer = []

                                    elif current_sequence == TARGET_SEQUENCE:
                                        print("ACCESS GRANTED! Kilit Acildi.")
                                        current_state = STATE_UNLOCKED
                                        current_sequence = []
                                        send_serial(b'UNLOCK\n')

                            # MOD 2: Yeni sifreyi belirleme
                            elif current_state == STATE_SETTING_PASS:
                                new_password_buffer.append(candidate_gesture)
                                print(f"Yeni Sifre Adimi: {candidate_gesture} ({len(new_password_buffer)}/4)")

                                if len(new_password_buffer) == 4:
                                    TARGET_SEQUENCE = new_password_buffer.copy()
                                    print(f"BASARILI! Yeni Sifre Kaydedildi: {TARGET_SEQUENCE}")
                                    current_state = STATE_AUTH
                                    new_password_buffer = []

                            last_gesture_time = current_time

                        current_gesture_frames = 0

            # Guvenlik zaman asimi
            if (current_state == STATE_AUTH and len(current_sequence) > 0) or \
                    (current_state == STATE_SETTING_PASS and len(new_password_buffer) > 0):
                if (time.time() - last_gesture_time) > sequence_timeout:
                    print("Zaman Asimi! Islem iptal edildi.")
                    current_sequence = []
                    new_password_buffer = []
                    current_state = STATE_AUTH

            # Okuma bari UI
            if SHOW_DISPLAY and candidate_gesture and current_gesture_frames > 0:
                cv2.putText(img,
                            f"Reading: {candidate_gesture} ({current_gesture_frames}/{REQUIRED_CONSECUTIVE_FRAMES})",
                            (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        elif current_state == STATE_UNLOCKED:
            if SHOW_DISPLAY:
                cv2.putText(img, "STATE: UNLOCKED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, "ACCESS GRANTED!", (150, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
                cv2.putText(img, "To lock: Step away from camera", (150, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

        # Ekran gosterimi
        if SHOW_DISPLAY:
            cv2.imshow("Air-Pass Security", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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