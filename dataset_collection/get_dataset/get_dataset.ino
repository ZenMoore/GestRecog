/**
   收集数据集
*/

#include <SoftwareSerial.h>

#define GEST_RX_PIN 19
#define GEST_TX_PIN 18
#define BUTTON 13

float data[6][2];
unsigned char Re_buf[11], counter = 0;
unsigned char sign = 0;
float a[3], angle[3];

//SoftwareSerial Serial1 = SoftwareSerial(GEST_TX_PIN, GEST_RX_PIN);

void setup() {
  Serial.begin(9600);

  Serial.println("initializing...");

  Serial1.begin(115200);
  Serial.println("baud rate：115200");

  byte baud[3] = {0xFF, 0xAA, 0x63};
  for (int i = 0; i < 3; i++) {
    Serial1.write(baud[i]);
  }
  Serial.println("baud rate 115200, return rate 100Hz.");
  Serial1.begin(115200);

  byte zzero[3] = {0xFF, 0xAA, 0x52};
  for (int i = 0; i < 3; i++) {
    Serial1.write(zzero[i]);
  } for (int i = 0; i < 3; i++) {
    Serial1.write(zzero[i]);
  }
  Serial.println("z-zeroing");

  byte acheck[3] = {0xFF, 0xAA, 0x67};
  for (int i = 0; i < 3; i++) {
    Serial1.write(acheck[i]);
  } for (int i = 0; i < 3; i++) {
    Serial1.write(acheck[i]);
  }
  Serial.println("A-calibration");

  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 256; j++) {
      data[i][j] = 0;
    }
  }

  pinMode(BUTTON, INPUT);
}

bool a_tuple() {
  int counter = 0;
  for (int i = 0; i < 6; i++) {
    if (data[i][sign] == 0) {
      counter++;
    }
  }

  if(counter >= 3){
    return false;
  }else{
    return true;
  }
}

void print_data() {
  for (int j = 0; j < 256; j++) {
    for (int i = 0; i < 6; i++) {
      Serial.print(data[i][j]);
      Serial.print("  ");
    }
    Serial.println();
  }
  Serial.println("===============================");
}

void initial() {
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 256; j++) {
      data[i][j] = 0;
    }
  }

  sign = 0;
}

void detect_and_store() {
  while (Serial1.available() && digitalRead(BUTTON) == HIGH) {
    Re_buf[counter] = (unsigned char)Serial1.read();
    if (counter == 0 && Re_buf[0] != 0x55) continue; //第0号数据不是帧头
    counter++;
    if (counter == 11)          //接收到11个数据
    {
      counter = 0;           //重新赋值，准备下一帧数据的接收
      if (Re_buf[0] == 0x55)   //检查帧头
      {
        switch (Re_buf [1])
        {
          case 0x51:
            data[0][sign] = (short(Re_buf [3] << 8 | Re_buf [2])) / 32768.0 * 16;
            data[1][sign] = (short(Re_buf [5] << 8 | Re_buf [4])) / 32768.0 * 16;
            data[2][sign] = (short(Re_buf [7] << 8 | Re_buf [6])) / 32768.0 * 16;
            if (a_tuple()) {
              sign++;
              if(sign >= 256){
                return;
              }
            }
            break;
          case 0x53:
            data[3][sign] = (short(Re_buf [3] << 8 | Re_buf [2])) / 32768.0 * 180;
            data[4][sign] = (short(Re_buf [5] << 8 | Re_buf [4])) / 32768.0 * 180;
            data[5][sign] = (short(Re_buf [7] << 8 | Re_buf [6])) / 32768.0 * 180;
            if (a_tuple()) {
              sign++;
              if(sign >= 256){
                return;
              }
            }
            break;
        }
      }
    }
  }
}
void loop() {
  if (digitalRead(BUTTON) == HIGH) {
    detect_and_store();
    print_data();
    initial();
  }
}
