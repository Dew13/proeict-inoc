void setup() 
{
  Serial.begin(9600);
}

void loop() 
{
  Serial.print("r");  //send the string "hello" and return the length of the string.
  delay(1000);
}
