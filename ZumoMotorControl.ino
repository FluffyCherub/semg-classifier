/*
 * This example uses the ZumoMotors library to drive each motor on the Zumo
 * forward, then backward. The yellow user LED is on when a motor should be
 * running forward and off when a motor should be running backward. If a
 * motor on your Zumo has been flipped, you can correct its direction by
 * uncommenting the call to flipLeftMotor() or flipRightMotor() in the setup()
 * function.
 */

#include <Wire.h>
#include <ZumoShield.h>

#define LED_PIN 13

ZumoMotors motors;
int x;

int speed = 200;

void setup()
{
  Serial.begin(115200);
  Serial.setTimeout(1);
  pinMode(LED_PIN, OUTPUT);

  // uncomment one or both of the following lines if your motors' directions need to be flipped
  //motors.flipLeftMotor(true);
  //motors.flipRightMotor(true);
}

void loop(){
 while (!Serial.available());
 x = Serial.readString().toInt();
 
 if(x == 1){
  // Stops or breaks
  motors.setLeftSpeed(0);
  motors.setRightSpeed(0);
  delay(500);
  Serial.print(x);        
 }
 else if(x == 2){
  // Right
  motors.setLeftSpeed(speed);
  motors.setRightSpeed(-speed);
  delay(500);
  Serial.print(x);  
 }
 else if(x == 3){
  // left
  motors.setLeftSpeed(-speed);
  motors.setRightSpeed(speed);
  delay(500);
  Serial.print(x);  
 }
 else if(x == 4){
  // down
  motors.setLeftSpeed(speed);
  motors.setRightSpeed(speed);
  delay(500);
  Serial.print(x);  
 }
 else if(x == 5){
  // up
  motors.setLeftSpeed(-speed);
  motors.setRightSpeed(-speed);
  delay(500);
  Serial.print(x);
 }
 else if(x == 0){
  motors.setLeftSpeed(0);
  motors.setRightSpeed(0);
  delay(500);
  Serial.print(x);
 }
 else{
  Serial.print("ERROR");
  motors.setLeftSpeed(0);
  motors.setRightSpeed(0);
 }
}
