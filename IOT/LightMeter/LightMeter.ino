#include <Wire.h>
#include <BH1750.h>

BH1750 lightMeter;

void setup() {
  Serial.begin(9600);
  Wire.begin();

  // Default address is usually 0x23
  if (lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE)) {
    Serial.println("BH1750 started");
  } else {
    Serial.println("Error initialising BH1750");
  }
}

void loop() {
  float lux = lightMeter.readLightLevel();

  // Serial.print("Light: ");
  Serial.println(lux);
  // Serial.println(" lx");

  delay(200);
}