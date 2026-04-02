#include <Wire.h>
#include <LiquidCrystal_PCF8574.h>

#define TRIG_PIN 5
#define ECHO_PIN 18
#define BUZZER 23
#define PULSE_PIN 34

LiquidCrystal_PCF8574 lcd(0x27);

// -------- VARIABLES --------
int occupancyCount = 0;

int pulseValue = 0;
int heartRate = 0;

unsigned long lastBeatTime = 0;
int threshold = 550; // adjust if needed

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

// -------- SETUP --------
void setup() {
  Serial.begin(115200);

  pinMode(PULSE_PIN, INPUT);
  
  pinMode(BUZZER, OUTPUT);
  digitalWrite(BUZZER, LOW);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  lcd.begin(16, 2);
  lcd.setBacklight(255);

  lcd.setCursor(0, 0);
  lcd.print("People Counter");
  delay(1000);
  lcd.clear();
}

// -------- LOOP --------
void loop() {

  long d = getFilteredDistance();

  // 🔥 SERIAL OUTPUT
  Serial.print("Distance: ");
  Serial.print(d);
  Serial.println(" cm");

  pulseValue = analogRead(PULSE_PIN);
  
  // detect beat
  if (pulseValue > threshold) {
    unsigned long currentTime = millis();
    
    if (currentTime - lastBeatTime > 300) { // debounce
      heartRate = 60000 / (currentTime - lastBeatTime);
      lastBeatTime = currentTime;
  
      Serial.print("BPM: ");
      Serial.println(heartRate);
  
      // 🚨 ALERT
      if (heartRate > 120 || heartRate < 50) {
        digitalWrite(BUZZER, HIGH);
      } else {
        digitalWrite(BUZZER, LOW);
      }
    }
  }

  if (Serial.available()) {
    String msg = Serial.readStringUntil('\n');
  
    if (msg == "ALERT") {
      digitalWrite(BUZZER, HIGH);
    } else {
      digitalWrite(BUZZER, LOW);
    }
  }
  
  if (d != -1) {
    updateHistory(d);

    if (bufferFilled && millis() - lastTriggerTime > cooldown) {

      // ENTER
      if (isIncreasing()) {
        occupancyCount++;

        Serial.println("ENTER detected");

        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print(">> ENTER >>");
        delay(500);

        lastTriggerTime = millis();
      }

      // EXIT
      else if (isDecreasing()) {
        occupancyCount--;

        if (occupancyCount < 0) occupancyCount = 0;

        Serial.println("EXIT detected");

        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print("<< EXIT <<");
        delay(500);

        lastTriggerTime = millis();
      }
    }
  }

  // LCD DISPLAY
  lcd.setCursor(0, 0);
  lcd.print("People: ");
  lcd.print(occupancyCount);
  lcd.print("   ");

  lcd.setCursor(0, 1);
  lcd.print("HR:");
  lcd.print(heartRate);
  lcd.print("   ");
  
  lcd.setCursor(0, 1);
  Serial.print("DIST:");
  Serial.print(d);
  
  Serial.print("|COUNT:");
  Serial.println(occupancyCount);
  delay(200);
}
