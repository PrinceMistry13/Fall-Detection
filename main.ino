void setup() {
  pinMode(8, OUTPUT);
  Serial.begin(9600);  // Must match Python
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == '1') {       // Fall detected
      digitalWrite(8, HIGH);
    }
    else if (command == '0') {  // No fall
      digitalWrite(8, LOW);
    }
  }
}