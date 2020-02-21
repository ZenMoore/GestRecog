/**
   收集数据集
*/
#define GEST_RX_PIN 19
#define GEST_TX_PIN 18
#define BUTTON 16

//预留400的空间，因为不知道JY61传来的数据有多少个，先用400存下来
float data[6][400];
//这三个变量是为获取JY61的数据而设置的临时变量
unsigned char Re_buf[11], counter = 0;
unsigned char sign = 0;
float a[3], angle[3];

void setup() {
  Serial.begin(9600);

  Serial.println("initializing...");

  //设置JY61初始波特率为 115200
  Serial1.begin(115200);
  Serial.println("baud rate：115200");

  //向JY61发送一次指令，确保其为115200的波特率
  byte baud[3] = {0xFF, 0xAA, 0x63};
  for (int i = 0; i < 3; i++) {
    Serial1.write(baud[i]);
  }
  Serial.println("baud rate 115200, return rate 100Hz.");
  Serial1.begin(115200);

  //下面是初始化步骤
  //特别注意：在执行此初始化时候，要保证JY61：x轴指向竖直向下，y轴指向水平向左，z轴指向水平向前！！！
  //z轴归零
  byte zzero[3] = {0xFF, 0xAA, 0x52};
  for (int i = 0; i < 3; i++) {
    Serial1.write(zzero[i]);
  } for (int i = 0; i < 3; i++) {
    Serial1.write(zzero[i]);
  }
  Serial.println("z-zeroing");

  //加计校准
  byte acheck[3] = {0xFF, 0xAA, 0x67};
  for (int i = 0; i < 3; i++) {
    Serial1.write(acheck[i]);
  } for (int i = 0; i < 3; i++) {
    Serial1.write(acheck[i]);
  }
  Serial.println("A-calibration");


  //初始化data数组
  initial();
  //设置按钮
  pinMode(BUTTON, INPUT);
}


// 由于JY61传数据时是一个包一个包的传送，一个包仅仅包含三个轴加速度或者三个轴角速度，因此加速度和角速度两类数据一次解包只能获取一类
// 这个函数用来判断data数组的第sign行即[6][sign]是否为完整的六列数据
// 判断方法是，如果出现多于三列数据值为0(初始值)，则是不完整数据（原则上，这并不是个完整的判断方法，但是足够用在咱们这个项目当中）
bool a_tuple() {
  int counter = 0;
  for (int i = 0; i < 6; i++) {
    if (data[i][sign] == 0) {
      counter++;
    }
  }

  if (counter >= 3) {
    return false;
  } else {
    return true;
  }
}

//在Serial中打印数据
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

//初始化data数组
void initial() {
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 256; j++) {
      data[i][j] = 0;
    }
  }
  sign = 0;
}

// 检测并读取JY61的数据，将读取到的数据存储在data数组当中
void detect_and_store() {
  while (digitalRead(BUTTON) == HIGH) {
    if (Serial1.available()) {
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
                if (sign >= 256) {
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
                if (sign >= 256) {
                  return;
                }
              }
              break;
          }
        }
      }
    }
  }
}


void loop() {
  if (digitalRead(BUTTON) == HIGH) {
    Serial.println("Button pushed.");
    detect_and_store();
    print_data();
    initial();
  }
}
