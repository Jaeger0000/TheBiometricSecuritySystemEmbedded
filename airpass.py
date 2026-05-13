import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import serial
import os
import urllib.request

# --- AYARLAR ---
ENABLE_SERIAL = False
SERIAL_PORT = 'COM3'
BAUD_RATE = 9600

# --- MODEL İNDİRİCİ ---
FACE_MODEL_PATH = 'blaze_face_short_range.tflite'
HAND_MODEL_PATH = 'hand_landmarker.task'

def download_model(url, filename):
    if not os.path.exists(filename):
        print(f"[{filename}] bulunamadı. İnternetten indiriliyor...")
        urllib.request.urlretrieve(url, filename)
        print(f"[{filename}] başarıyla indirildi!")

download_model("https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite", FACE_MODEL_PATH)
download_model("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", HAND_MODEL_PATH)

# --- SERİ HABERLEŞME KURULUMU ---
if ENABLE_SERIAL:
    try:
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("Arduino'ya bağlanıldı.")
    except Exception as e:
        print(f"Seri port hatası: {e}")
        ENABLE_SERIAL = False

# --- MEDIAPIPE TASKS API KURULUMU ---
face_base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
face_options = vision.FaceDetectorOptions(base_options=face_base_options)
face_detector = vision.FaceDetector.create_from_options(face_options)

hand_base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
hand_options = vision.HandLandmarkerOptions(base_options=hand_base_options, num_hands=1)
hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

# ==========================================
# --- SİSTEM DURUMLARI (STATE MACHINE) ---
# ==========================================
STATE_IDLE = 0          # Kilitli, yüz yok
STATE_AUTH = 1          # Yüz var, kilitli, şifre veya admin komutu bekliyor
STATE_UNLOCKED = 2      # Kilit açık (Yüz ekranda olduğu sürece)
STATE_SETTING_PASS = 3  # Admin modu: Yeni 4'lü şifre bekleniyor

current_state = STATE_IDLE

# Şifre Değişkenleri
TARGET_SEQUENCE = ["Fist", "Peace", "Open", "Fist"] # Varsayılan 4'lü kilit açma şifresi
ADMIN_SEQUENCE = ["Fist", "Open", "Fist"]           # Şifre belirleme modunu tetikleyen komut
current_sequence = []
new_password_buffer = []

# Zamanlama ve Debounce Değişkenleri
last_gesture_time = 0
sequence_timeout = 5.0
gesture_cooldown = 1.5
REQUIRED_CONSECUTIVE_FRAMES = 10
current_gesture_frames = 0
candidate_gesture = None

def get_gesture(landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Başparmak
    if landmarks[tip_ids[0]].x > landmarks[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Diğer Parmaklar
    for i in range(1, 5):
        if landmarks[tip_ids[i]].y < landmarks[tip_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # Jestleri Sınıflandır
    if fingers == [0, 0, 0, 0, 0] or fingers == [1, 0, 0, 0, 0]:
        return "Fist"
    elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
        return "Peace"
    elif fingers == [1, 1, 1, 1, 1] or fingers[1:] == [1, 1, 1, 1]:
        return "Open"

    return "Unknown"

# --- ANA DÖNGÜ ---
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    h, w, c = img.shape

    # ==========================================
    # 1. YÜZ ALGILAMA VE OTOMATİK KİLİTLEME
    # ==========================================
    face_result = face_detector.detect(mp_image)

    if len(face_result.detections) > 0:
        if current_state == STATE_IDLE:
            current_state = STATE_AUTH
            print("Yüz Algılandı. Sistem Aktif.")

        # Yüz Çerçevesinin Rengini Duruma Göre Belirle
        # UNLOCKED ise Yeşil, diğer tüm durumlarda Kırmızı
        bbox_color = (0, 255, 0) if current_state == STATE_UNLOCKED else (0, 0, 255)

        for detection in face_result.detections:
            bbox = detection.bounding_box
            cv2.rectangle(img, (bbox.origin_x, bbox.origin_y),
                          (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), bbox_color, 2)
    else:
        # YÜZ KAYBOLDUĞU AN SİSTEMİ KİLİTLE (Sürekli Kimlik Doğrulama)
        if current_state != STATE_IDLE:
            print("Yüz kayboldu! Sistem otomatik olarak kilitlendi.")
            current_state = STATE_IDLE
            current_sequence = []
            new_password_buffer = []
            current_gesture_frames = 0
            if ENABLE_SERIAL:
                arduino.write(b'LOCK\n')

    # ==========================================
    # 2. DURUM MAKİNESİ (STATE HANDLING)
    # ==========================================

    if current_state == STATE_IDLE:
        cv2.putText(img, "SYSTEM LOCKED - NO FACE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    elif current_state == STATE_AUTH or current_state == STATE_SETTING_PASS:
        # Ekran Bilgileri
        if current_state == STATE_AUTH:
            cv2.putText(img, "STATE: LOCKED (Waiting for Command)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(img, f"Seq: {' -> '.join(current_sequence)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(img, "STATE: ADMIN (Setting New Passcode)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(img, f"New Pass: {' -> '.join(new_password_buffer)} (Need 4)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # El Algılama
        hand_result = hand_landmarker.detect(mp_image)
        if len(hand_result.hand_landmarks) > 0:
            landmarks = hand_result.hand_landmarks[0]
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 4, (0, 255, 255), cv2.FILLED)

            detected_gesture = get_gesture(landmarks)

            # --- DEBOUNCE MANTIĞI ---
            if detected_gesture != "Unknown":
                if detected_gesture == candidate_gesture:
                    current_gesture_frames += 1
                else:
                    candidate_gesture = detected_gesture
                    current_gesture_frames = 1

                if current_gesture_frames >= REQUIRED_CONSECUTIVE_FRAMES:
                    current_time = time.time()

                    if (current_time - last_gesture_time) > gesture_cooldown:

                        # --- MOD 1: ŞİFRE VEYA ADMIN KOMUTU GİRİŞİ ---
                        if current_state == STATE_AUTH:
                            temp_seq = current_sequence + [candidate_gesture]

                            # Eşleşme Kontrolü (Girilen seri, Admin veya Target şifresinin bir parçası mı?)
                            is_admin_path = (ADMIN_SEQUENCE[:len(temp_seq)] == temp_seq)
                            is_target_path = (TARGET_SEQUENCE[:len(temp_seq)] == temp_seq)

                            if not (is_admin_path or is_target_path):
                                print(f"Yanlış Hareket ({candidate_gesture}). Dizi Sıfırlandı.")
                                current_sequence = []
                            else:
                                current_sequence.append(candidate_gesture)
                                print(f"Adım Başarılı: {candidate_gesture}. Mevcut Durum: {current_sequence}")

                                # ADMIN KOMUTU TAMAMLANDI MI?
                                if current_sequence == ADMIN_SEQUENCE:
                                    print("--- ADMIN MODU AKTİF --- Yeni 4'lü şifrenizi girin.")
                                    current_state = STATE_SETTING_PASS
                                    current_sequence = []
                                    new_password_buffer = []

                                # KİLİT AÇMA ŞİFRESİ TAMAMLANDI MI?
                                elif current_sequence == TARGET_SEQUENCE:
                                    print("ACCESS GRANTED! Kilit Açıldı.")
                                    current_state = STATE_UNLOCKED
                                    current_sequence = []
                                    if ENABLE_SERIAL:
                                        arduino.write(b'UNLOCK\n')

                        # --- MOD 2: YENİ ŞİFREYİ BELİRLEME ---
                        elif current_state == STATE_SETTING_PASS:
                            new_password_buffer.append(candidate_gesture)
                            print(f"Yeni Şifre Adımı eklendi: {candidate_gesture} ({len(new_password_buffer)}/4)")

                            if len(new_password_buffer) == 4:
                                TARGET_SEQUENCE = new_password_buffer.copy()
                                print(f"BAŞARILI! Yeni Şifreniz Kaydedildi: {TARGET_SEQUENCE}")
                                current_state = STATE_AUTH # Şifre belirlendikten sonra kilitli duruma dön
                                new_password_buffer = []

                        last_gesture_time = current_time

                    current_gesture_frames = 0

        # --- GÜVENLİK ZAMAN AŞIMI KONTROLÜ ---
        if (current_state == STATE_AUTH and len(current_sequence) > 0) or (current_state == STATE_SETTING_PASS and len(new_password_buffer) > 0):
            if (time.time() - last_gesture_time) > sequence_timeout:
                print("Zaman Aşımı! İşlem iptal edildi ve sıfırlandı.")
                current_sequence = []
                new_password_buffer = []
                current_state = STATE_AUTH

        # Okuma Barı UI
        if candidate_gesture and current_gesture_frames > 0:
            cv2.putText(img, f"Reading: {candidate_gesture} ({current_gesture_frames}/{REQUIRED_CONSECUTIVE_FRAMES})", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    elif current_state == STATE_UNLOCKED:
        cv2.putText(img, "STATE: UNLOCKED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, "ACCESS GRANTED!", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
        cv2.putText(img, "To lock: Step away from camera", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    cv2.imshow("Air-Pass Security", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if ENABLE_SERIAL:
    arduino.close()