int x;
void setup() {
 Serial.begin(115200);
 Serial.setTimeout(1);
 pinMode(11, OUTPUT);
}
void loop() {
 while (!Serial.available());
 x = Serial.readString().toInt();
 if (x == 1){
  digitalWrite(11, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(1000);              // wait for a second
  digitalWrite(11, LOW);    // turn the LED off by making the voltage LOW
  delay(1000);              // wait for a second
 }
 Serial.print(x + 1);
}
