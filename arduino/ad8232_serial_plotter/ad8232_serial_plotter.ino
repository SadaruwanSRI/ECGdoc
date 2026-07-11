/*
  AD8232 + classic Arduino Nano: standalone ECG acquisition test

  Serial Plotter output:
    ECG:<0..1023> LeadOff:<0 or 1023> Min:0 Max:1023

  Wiring:
    AD8232 3.3V   -> Nano 3V3
    AD8232 GND    -> Nano GND
    AD8232 OUTPUT -> Nano A0
    AD8232 LO+    -> Nano D10
    AD8232 LO-    -> Nano D11
    AD8232 SDN    -> not connected

  The sketch samples at approximately 128 Hz using micros(), matching the
  sampling rate expected by the ECG analysis application.
*/

const uint8_t ECG_PIN = A0;
const uint8_t LO_PLUS_PIN = 10;
const uint8_t LO_MINUS_PIN = 11;

const unsigned long SERIAL_BAUD = 115200;
const unsigned long SAMPLE_INTERVAL_US = 7813; // approximately 1 / 128 second

unsigned long nextSampleUs;

void setup() {
  pinMode(ECG_PIN, INPUT);
  pinMode(LO_PLUS_PIN, INPUT);
  pinMode(LO_MINUS_PIN, INPUT);
  pinMode(LED_BUILTIN, OUTPUT);

  Serial.begin(SERIAL_BAUD);
  nextSampleUs = micros();
}

void loop() {
  const unsigned long now = micros();

  // Signed subtraction keeps this comparison safe when micros() wraps.
  if ((long)(now - nextSampleUs) < 0) {
    return;
  }

  nextSampleUs += SAMPLE_INTERVAL_US;

  // If serial output or another delay made us fall far behind, restart the
  // schedule instead of rapidly emitting a burst of stale samples.
  if ((long)(now - nextSampleUs) > (long)SAMPLE_INTERVAL_US) {
    nextSampleUs = now + SAMPLE_INTERVAL_US;
  }

  const bool leadOff = digitalRead(LO_PLUS_PIN) == HIGH ||
                       digitalRead(LO_MINUS_PIN) == HIGH;
  const int ecg = analogRead(ECG_PIN);

  digitalWrite(LED_BUILTIN, leadOff ? HIGH : LOW);

  // Named, tab-separated values are understood by Arduino IDE Serial Plotter.
  // Min and Max keep the vertical scale fixed so changes are easy to compare.
  Serial.print("ECG:");
  Serial.print(ecg);
  Serial.print('\t');
  Serial.print("LeadOff:");
  Serial.print(leadOff ? 1023 : 0);
  Serial.print('\t');
  Serial.print("Min:0\tMax:1023\n");
}
