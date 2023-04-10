import serial
import time


name = "XX"  # 当前测试者编号

# 打开串口连接
ser = serial.Serial('COM3', 9600)
while True:
    # 10s信号稳定期
    print('10秒稳定期！')
    start_time = time.time()
    while True:
        line = ser.readline().decode('utf-8').rstrip()
        if time.time() - start_time > 10:
            break
    i = 1
    while i < 6:

        print("请做出%d姿势！" % i)

        # 10s稳定期
        print('10秒稳定期')
        start_time = time.time()
        while True:
            line = ser.readline().decode('utf-8').rstrip()
            if time.time() - start_time > 10:
                break

        # 开始保存数据
        fp = 'datas/' + name + str(i) + '.txt'
        with open(fp, 'w', encoding='utf-8') as f:
            print('开始采样%d姿势数据信号！' % i)
            start_time = time.time()
            while True:
                # 从串口读取数据
                line = ser.readline().decode('utf-8').rstrip()

                # 将读取的数据分解成每个引脚的值
                values = line.split(',')

                # 如果读取到了所有引脚的值，则输出它们
                if len(values) == 5:
                    valA0 = values[0].split(':')[1]
                    valA1 = values[1].split(':')[1]
                    valA2 = values[2].split(':')[1]
                    valA3 = values[3].split(':')[1]
                    valA4 = values[4].split(':')[1]

                    f.write(valA0)
                    f.write(";")
                    f.write(valA1)
                    f.write(";")
                    f.write(valA2)
                    f.write(";")
                    f.write(valA3)
                    f.write(";")
                    f.write(valA4)
                    f.write(";")
                    f.write(str(i))
                    f.write("\n")

                if time.time() - start_time > 15:
                    print("%d的姿势训练数据采集完毕！" % i)
                    break

        fp = 'validdatas/' + name + str(i) + '.txt'
        with open(fp, 'w', encoding='utf-8') as f:
            start_time = time.time()
            while True:
                # 从串口读取数据
                line = ser.readline().decode('utf-8').rstrip()

                # 将读取的数据分解成每个引脚的值
                values = line.split(',')

                # 如果读取到了所有引脚的值，则输出它们
                if len(values) == 5:
                    valA0 = values[0].split(':')[1]
                    valA1 = values[1].split(':')[1]
                    valA2 = values[2].split(':')[1]
                    valA3 = values[3].split(':')[1]
                    valA4 = values[4].split(':')[1]

                    f.write(valA0)
                    f.write(";")
                    f.write(valA1)
                    f.write(";")
                    f.write(valA2)
                    f.write(";")
                    f.write(valA3)
                    f.write(";")
                    f.write(valA4)
                    f.write(";")
                    f.write(str(i))
                    f.write("\n")

                if time.time() - start_time > 5:
                    print("%d的姿势测试数据采集完毕！" % i)
                    break
        i += 1

    print("数据采集结束")

    while True:
        # 从串口读取数据
        line = ser.readline().decode('utf-8').rstrip()

        # 将读取的数据分解成每个引脚的值
        values = line.split(',')

        # 如果读取到了所有引脚的值，则输出它们
        if len(values) == 5:
            valA0 = values[0].split(':')[1]
            valA1 = values[1].split(':')[1]
            valA2 = values[2].split(':')[1]
            valA3 = values[3].split(':')[1]
            valA4 = values[4].split(':')[1]

        print("A0:", valA0, "A1:", valA1, "A2:", valA2, "A3:", valA3, "A4:", valA4)

