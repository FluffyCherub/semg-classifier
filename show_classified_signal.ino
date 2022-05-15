int x;
void setup() {
 Serial.begin(115200);
 Serial.setTimeout(1);
 pinMode(9, OUTPUT);
 pinMode(10, OUTPUT);
 pinMode(11, OUTPUT);
 pinMode(12, OUTPUT);
 pinMode(13, OUTPUT);
}

void clear(){
  digitalWrite(9, LOW);
  digitalWrite(10, LOW);
  digitalWrite(11, LOW);
  digitalWrite(12, LOW);
  digitalWrite(13, LOW);
}

void loop() {
 while (!Serial.available());
 x = Serial.readString().toInt();
 
 if (x == 1){
  // Fist
  clear();
  digitalWrite(9, HIGH);
  Serial.print(x);        
 }
 else if(x == 2){
  // right
  clear();
  digitalWrite(10, HIGH);
  Serial.print(x);  
 }
 else if(x == 3){
  // left
  clear();
  digitalWrite(11, HIGH);
  Serial.print(x);  
 }
 else if(x == 4){
  // down
  clear();
  digitalWrite(12, HIGH);
  Serial.print(x);  
 }
 else if(x == 5){
  // up
  clear();
  digitalWrite(13, HIGH);
  Serial.print(x);  
 }
 else if(x == 0){
  // rest
  clear();
  Serial.print(x);  
 }
 else{
  clear();
  digitalWrite(9, HIGH);
  digitalWrite(10, HIGH);
  digitalWrite(11, HIGH);
  digitalWrite(12, HIGH);
  digitalWrite(13, HIGH);
  Serial.print("ERROR");  
 }
 
}
