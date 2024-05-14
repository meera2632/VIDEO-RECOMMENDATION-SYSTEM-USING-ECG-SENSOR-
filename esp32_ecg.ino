/*
  ESP32 + AD8232 ECG data collector.
  Sends raw ADC values to Ubidots over MQTT.
  Replace the placeholder credentials locally; do not commit real secrets.
*/
#include <WiFi.h>
#include <PubSubClient.h>

const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* UBIDOTS_TOKEN = "YOUR_UBIDOTS_TOKEN";
const char* DEVICE_LABEL = "esp32_ecg";
const char* VARIABLE_LABEL = "ecg";
const char* MQTT_BROKER = "industrial.api.ubidots.com";
const int MQTT_PORT = 1883;
const int ECG_PIN = 34;
const int LO_PLUS_PIN = 32;
const int LO_MINUS_PIN = 33;
const unsigned long SAMPLE_INTERVAL_US = 4000; // 250 Hz

WiFiClient wifiClient;
PubSubClient mqttClient(wifiClient);
unsigned long nextSample = 0;
char payload[180];
char topic[120];

void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) delay(500);
}

void connectMQTT() {
  while (!mqttClient.connected()) {
    String clientId = "ecg-" + String((uint32_t)ESP.getEfuseMac(), HEX);
    mqttClient.connect(clientId.c_str(), UBIDOTS_TOKEN, "");
    if (!mqttClient.connected()) delay(2000);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(ECG_PIN, INPUT);
  pinMode(LO_PLUS_PIN, INPUT);
  pinMode(LO_MINUS_PIN, INPUT);
  analogReadResolution(12);
  connectWiFi();
  mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
  snprintf(topic, sizeof(topic), "/v1.6/devices/%s", DEVICE_LABEL);
  nextSample = micros();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();
  if (!mqttClient.connected()) connectMQTT();
  mqttClient.loop();

  unsigned long now = micros();
  if ((long)(now - nextSample) >= 0) {
    nextSample += SAMPLE_INTERVAL_US;
    bool leadsOff = digitalRead(LO_PLUS_PIN) || digitalRead(LO_MINUS_PIN);
    int raw = analogRead(ECG_PIN);
    snprintf(
      payload, sizeof(payload),
      "{\"%s\":{\"value\":%d,\"leads_off\":%d}}",
      VARIABLE_LABEL, raw, leadsOff ? 1 : 0
    );
    mqttClient.publish(topic, payload);
  }
}

