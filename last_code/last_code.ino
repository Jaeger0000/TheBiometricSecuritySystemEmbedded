#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>
#include <SPI.h>
#include <Servo.h>

#define TFT_CS    10
#define TFT_RST   8
#define TFT_DC    9

// Yazılımsal SPI için veri ve saat pinleri
#define TFT_MOSI  11
#define TFT_SCLK  13

// ST7735'i yazılımsal SPI ile başlatıyoruz
Adafruit_ST7735 tft = Adafruit_ST7735(TFT_CS, TFT_DC, TFT_MOSI, TFT_SCLK, TFT_RST);

String rxLine = "";
String line1 = "Booting...";
String line2 = "Waiting data";
String line3 = "";
String line4 = "";
String lastLine1 = "";
String lastLine2 = "";
String lastLine3 = "";
String lastLine4 = "";

// Servo and LED pins
const int SERVO_PIN = 3;
const int GREEN_LED_PIN = 5;
const int RED_LED_PIN = 6;

Servo lockServo;
const int LOCK_ANGLE = 0;    // Kilitli açı
const int UNLOCK_ANGLE = 90; // Açık açı
bool isLocked = false;       // Başlangıçta kilit fonksiyonunu tetiklemek için false başlattık


const int LINE_X = 2;
const int LINE1_Y = 6;
const int LINE2_Y = 24;
const int LINE3_Y = 42;
const int LINE4_Y = 60;

const int LINE_W = 160;
const int LINE_H = 18; 

void drawScreen();

void showStartupScreen() {
  line1 = "Hello Embedded";
  line2 = "Waiting Python";
  line3 = "";
  line4 = "";
  drawScreen();
}

void drawScreen() {
  tft.setTextColor(ST77XX_WHITE, ST77XX_BLACK); 
  tft.setTextSize(1); 

  tft.fillRect(0, LINE1_Y - 2, LINE_W, LINE_H, ST77XX_BLACK);
  tft.setCursor(LINE_X, LINE1_Y);
  tft.print(line1);
  lastLine1 = line1;

  tft.fillRect(0, LINE2_Y - 2, LINE_W, LINE_H, ST77XX_BLACK);
  tft.setCursor(LINE_X, LINE2_Y);
  tft.print(line2);
  lastLine2 = line2;

  tft.fillRect(0, LINE3_Y - 2, LINE_W, LINE_H, ST77XX_BLACK);
  tft.setCursor(LINE_X, LINE3_Y);
  tft.print(line3);
  lastLine3 = line3;

  tft.fillRect(0, LINE4_Y - 2, LINE_W, LINE_H, ST77XX_BLACK);
  tft.setCursor(LINE_X, LINE4_Y);
  tft.print(line4);
  lastLine4 = line4;
}

String getField(const String &data, int index) {
  int start = 0;
  int field = 0;

  for (int i = 0; i <= data.length(); i++) {
    if (i == data.length() || data.charAt(i) == '|') {
      if (field == index) {
        return data.substring(start, i);
      }
      field++;
      start = i + 1;
    }
  }

  return "";
}

void processCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) {
    return;
  }

  if (cmd.startsWith("STATUS|")) {
    line1 = getField(cmd, 1);
    line2 = getField(cmd, 2);
    line3 = getField(cmd, 3);
    line4 = getField(cmd, 4);
    drawScreen();
    return;
  }

  if (cmd == "LOCK") {
    setLockState(true);
    return;
  }

  if (cmd == "UNLOCK") {
    if (isLocked) {
      line1 = "STATE: UNLOCKED";
      line2 = "Access granted!";
      line3 = "Welcome...";
      line4 = "Door is open";
      drawScreen();
      setLockState(false);
    }
    return;
  }
}

void setLockState(bool lock) {
  if (lock == isLocked) return; // no change
  isLocked = lock;
  if (isLocked) {
    // move servo to locked position and light red LED
    lockServo.write(LOCK_ANGLE);
    digitalWrite(RED_LED_PIN, HIGH);
    digitalWrite(GREEN_LED_PIN, LOW);
  } else {
    // move servo to unlocked position and light green LED
    lockServo.write(UNLOCK_ANGLE);
    digitalWrite(RED_LED_PIN, LOW);
    digitalWrite(GREEN_LED_PIN, HIGH);
  }
  delay(300); // give servo time to move
}

void setup() {
  Serial.begin(115200);

  // STARTUP: Direk Kırmızı LED'i yakıyoruz
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(GREEN_LED_PIN, OUTPUT);
  digitalWrite(RED_LED_PIN, HIGH);
  digitalWrite(GREEN_LED_PIN, LOW);

  // Initialize servo
  lockServo.attach(SERVO_PIN);
  
  // Start locked by default (Artık açılışta servoyu fiziksel olarak da kilit yönüne itecek)
  setLockState(true);

  tft.initR(INITR_MINI160x80); 
  tft.invertDisplay(true);
  tft.setRotation(3);

  tft.fillScreen(ST77XX_BLACK);
  showStartupScreen();
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();

    if (c == '\n') {
      processCommand(rxLine);
      rxLine = "";
    } else if (c != '\r') {
      rxLine += c;
      if (rxLine.length() > 200) {
        rxLine = "";
      }
    }
  }
}