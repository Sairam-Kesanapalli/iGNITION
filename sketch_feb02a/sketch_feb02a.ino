#include <Wire.h>
#include <LiquidCrystal_PCF8574.h>
#include <DHT.h>

#define TRIG_PIN 5
#define ECHO_PIN 18
#define BUZZER 25
#define DHT_PIN 4
#define DHT_TYPE DHT11
#define MQ7_PIN 34

// Buzzer modes:
// 0 = passive piezo (tone only)
// 1 = active buzzer, active HIGH
// 2 = active buzzer, active LOW
// 3 = dual drive (tone + HIGH) for unknown buzzer type/wiring tests
const uint8_t BUZZER_MODE = 3;
const uint16_t BUZZER_TONE_HZ = 2200;

LiquidCrystal_PCF8574 lcd(0x27);
bool lcdReady = false;
DHT dht(DHT_PIN, DHT_TYPE);

// -------- VARIABLES --------
int occupancyCount = 0;

bool serverAlertActive = false;
unsigned long serverAlertUntil = 0;
const unsigned long serverAlertHoldMs = 3000;

bool buzzerState = false;
bool envAlertLatched = false;
bool tempAlertActive = false;
bool alcoholAlertActive = false;

float lastTempC = NAN;
float lastHumidity = NAN;
int lastMQ7Value = 0;
unsigned long lastSensorReadMs = 0;
const unsigned long sensorReadIntervalMs = 1200;

// Safety thresholds (tune after field tests)
const float TEMP_ALERT_C = 36.5;
const int MQ7_ALERT_RAW = 1700;
const bool MQ7_ENABLED = false;
const uint8_t MQ7_HIGH_CONFIRM_COUNT = 4;
uint8_t mq7HighCount = 0;

long history[5] = {0};
int histIndex = 0;
bool bufferFilled = false;

unsigned long lastTriggerTime = 0;
const int cooldown = 2000;

// -------- DISTANCE --------
long getDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  if (duration == 0) return -1;

  long d = duration * 0.034 / 2;
  if (d < 2 || d > 400) return -1;

  return d;
}

// -------- FILTER --------
long getFilteredDistance() {
  long readings[5];
  int count = 0;

  for (int i = 0; i < 5; i++) {
    long d = getDistance();
    if (d != -1) readings[count++] = d;
    delay(5);
  }

  if (count == 0) return -1;

  for (int i = 0; i < count - 1; i++) {
    for (int j = i + 1; j < count; j++) {
      if (readings[i] > readings[j]) {
        long t = readings[i];
        readings[i] = readings[j];
        readings[j] = t;
      }
    }
  }

  return readings[count / 2];
}

// -------- TREND --------
void updateHistory(long d) {
  history[histIndex] = d;
  histIndex = (histIndex + 1) % 5;
  if (histIndex == 0) bufferFilled = true;
}

bool isIncreasing() {
  for (int i = 0; i < 4; i++)
    if (history[i] >= history[i + 1]) return false;
  return true;
}

bool isDecreasing() {
  for (int i = 0; i < 4; i++)
    if (history[i] <= history[i + 1]) return false;
  return true;
}

void setBuzzer(bool on) {
  if (buzzerState == on) return;
  buzzerState = on;

  if (BUZZER_MODE == 0) {
    if (on) tone(BUZZER, BUZZER_TONE_HZ);
    else noTone(BUZZER);
  } else if (BUZZER_MODE == 1) {
    noTone(BUZZER);
    digitalWrite(BUZZER, on ? HIGH : LOW);
  } else if (BUZZER_MODE == 2) {
    noTone(BUZZER);
    digitalWrite(BUZZER, on ? LOW : HIGH);
  } else {
    if (on) {
      tone(BUZZER, BUZZER_TONE_HZ);
      digitalWrite(BUZZER, HIGH);
    } else {
      noTone(BUZZER);
      digitalWrite(BUZZER, LOW);
    }
  }

  Serial.print("BUZZER:");
  Serial.print(on ? "ON" : "OFF");
  Serial.print(" MODE:");
  Serial.println(BUZZER_MODE);
}

void buzzerSelfTest() {
  // Audible startup check so wiring/type problems are obvious immediately.
  setBuzzer(true);
  delay(350);
  setBuzzer(false);
  delay(150);
  setBuzzer(true);
  delay(350);
  setBuzzer(false);
}

void readEnvironmentSensors() {
  unsigned long now = millis();
  if (now - lastSensorReadMs < sensorReadIntervalMs) {
    return;
  }
  lastSensorReadMs = now;

  float t = dht.readTemperature();
  float h = dht.readHumidity();
  int mq7 = analogRead(MQ7_PIN);

  if (!isnan(t)) lastTempC = t;
  if (!isnan(h)) lastHumidity = h;
  lastMQ7Value = mq7;

  tempAlertActive = (!isnan(lastTempC) && lastTempC >= TEMP_ALERT_C);
  if (MQ7_ENABLED) {
    if (lastMQ7Value >= MQ7_ALERT_RAW) {
      if (mq7HighCount < 255) mq7HighCount++;
    } else {
      mq7HighCount = 0;
    }
    alcoholAlertActive = (mq7HighCount >= MQ7_HIGH_CONFIRM_COUNT);
  } else {
    mq7HighCount = 0;
    alcoholAlertActive = false;
  }

  if ((tempAlertActive || alcoholAlertActive) && occupancyCount > 0) {
    envAlertLatched = true;
  }

  if (occupancyCount <= 0) {
    envAlertLatched = false;
  }

  Serial.print("ENV:T=");
  if (isnan(lastTempC)) Serial.print("nan");
  else Serial.print(lastTempC, 1);
  Serial.print("|H=");
  if (isnan(lastHumidity)) Serial.print("nan");
  else Serial.print(lastHumidity, 1);
  Serial.print("|MQ7=");
  Serial.print(lastMQ7Value);
  Serial.print("|TEMP_ALERT=");
  Serial.print(tempAlertActive ? 1 : 0);
  Serial.print("|ALC_ALERT=");
  Serial.print(alcoholAlertActive ? 1 : 0);
  Serial.print("|ALC_HCNT=");
  Serial.print(mq7HighCount);
  Serial.print("|LATCH=");
  Serial.println(envAlertLatched ? 1 : 0);
}

bool i2cDevicePresent(uint8_t address) {
  Wire.beginTransmission(address);
  return Wire.endTransmission() == 0;
}

// -------- SETUP --------
void setup() {
  Serial.begin(74880);
  
  pinMode(BUZZER, OUTPUT);
  digitalWrite(BUZZER, LOW);
  noTone(BUZZER);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(MQ7_PIN, INPUT);
  dht.begin();

  Wire.begin();
  if (i2cDevicePresent(0x27)) {
    lcd.begin(16, 2);
    lcd.setBacklight(255);
    lcd.setCursor(0, 0);
    lcd.print("People Counter");
    delay(1000);
    lcd.clear();
    lcdReady = true;
    Serial.println("LCD:READY");
  } else {
    lcdReady = false;
    Serial.println("LCD:NOT_FOUND (running without LCD)");
  }

  Serial.print("READY BUZZER_PIN=");
  Serial.print(BUZZER);
  Serial.print(" MODE=");
  Serial.println(BUZZER_MODE);

  buzzerSelfTest();
}

// -------- LOOP --------
void loop() {

  long d = getFilteredDistance();
  readEnvironmentSensors();

  // 🔥 SERIAL OUTPUT
  Serial.print("Distance: ");
  Serial.print(d);
  Serial.println(" cm");

  if (Serial.available()) {
    String msg = Serial.readStringUntil('\n');
    msg.trim();
  
    if (msg.startsWith("ALERT")) {
      Serial.print("CMD:");
      Serial.println(msg);
      serverAlertActive = true;
      serverAlertUntil = millis() + serverAlertHoldMs;
    } else if (msg == "BUZZER_TEST") {
      Serial.println("CMD:BUZZER_TEST");
      buzzerSelfTest();
    } else if (msg == "BUZZER_ON") {
      Serial.println("CMD:BUZZER_ON");
      setBuzzer(true);
      delay(1200);
      setBuzzer(false);
    } else if (msg == "BUZZER_PULSE") {
      Serial.println("CMD:BUZZER_PULSE");
      for (int i = 0; i < 8; i++) {
        setBuzzer(true);
        delay(120);
        setBuzzer(false);
        delay(120);
      }
    } else if (msg == "OK") {
      Serial.println("CMD:OK");
      serverAlertActive = false;
      serverAlertUntil = 0;
    }
  }

  if (serverAlertActive && millis() > serverAlertUntil) {
    serverAlertActive = false;
  }

  bool buzzerShouldBeOn = serverAlertActive || envAlertLatched;
  if (buzzerShouldBeOn) {
    setBuzzer(true);
  } else {
    setBuzzer(false);
  }
  
  if (d != -1) {
    updateHistory(d);

    if (bufferFilled && millis() - lastTriggerTime > cooldown) {
      // ENTER
      if (isIncreasing()) {
        occupancyCount++;

        Serial.println("ENTER detected");
        Serial.print("COUNT:");
        Serial.print(occupancyCount);
        Serial.println(",DOOR:MAIN,EVENT:ENTER");

        if (lcdReady) {
          lcd.clear();
          lcd.setCursor(0, 0);
          lcd.print(">> ENTER >>");
        }
        delay(500);

        lastTriggerTime = millis();
      }

      // EXIT
      else if (isDecreasing()) {
        occupancyCount--;

        if (occupancyCount < 0) occupancyCount = 0;

        Serial.println("EXIT detected");
        Serial.print("COUNT:");
        Serial.print(occupancyCount);
        Serial.println(",DOOR:MAIN,EVENT:EXIT");

        if (lcdReady) {
          lcd.clear();
          lcd.setCursor(0, 0);
          lcd.print("<< EXIT <<");
        }
        delay(500);

        lastTriggerTime = millis();
      }

      if (occupancyCount <= 0) {
        envAlertLatched = false;
      }
    }
  }

  // LCD DISPLAY
  if (lcdReady) {
    lcd.setCursor(0, 0);
    if (envAlertLatched) {
      if (tempAlertActive && alcoholAlertActive) {
        lcd.print("ALERT:TEMP+ALC ");
      } else if (tempAlertActive) {
        lcd.print("ALERT:TEMP HI ");
      } else if (alcoholAlertActive) {
        lcd.print("ALERT:ALCOHOL ");
      } else {
        lcd.print("ALERT:CHECK BUS");
      }
    } else {
      lcd.print("People: ");
      lcd.print(occupancyCount);
      lcd.print("   ");
    }

    lcd.setCursor(0, 1);
    if (!isnan(lastTempC)) {
      lcd.print("T:");
      lcd.print(lastTempC, 1);
    } else {
      lcd.print("T:na");
    }
    if (MQ7_ENABLED) {
      lcd.print(" MQ:");
      lcd.print(lastMQ7Value);
      lcd.print(" ");
    } else {
      lcd.print(" D:");
      lcd.print(d);
      lcd.print("   ");
    }
  }
  
  Serial.print("DIST:");
  Serial.print(d);
  
  Serial.print("|COUNT:");
  Serial.println(occupancyCount);
  delay(200);
}
