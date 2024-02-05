from scipy import linalg
import numpy as np
from numpy import loadtxt
import math
from Dtreefunc import *

f = open("preprocessed_iris.txt", "r")
X = f.readlines()

# เตรียม พื้นที่เพื่อเก็บผลการคำนวณแยกตาม class
N = 3  # col(3 class)
M = 3  # row

sl = np.zeros(3)
slCI = [[0 for i in range(M)] for j in range(N)]

sw = np.zeros(3)
swCI = [[0 for i in range(M)] for j in range(N)]

pl = np.zeros(3)
plCI = [[0 for i in range(M)] for j in range(N)]

pw = np.zeros(3)
pwCI = [[0 for i in range(M)] for j in range(N)]

classs = np.zeros(3)

# วน loop เพื่อนับข้อมูล แยกตามรายละเอียด attb และ class
for i in range(0, 15):
    if X[i].count("S") == 1:
        sl[0] += 1  
        if ((X[i].count("S") == 1)) and (X[i].count("Iris-setosa") == 1):
            slCI[0][0] += 1  
        elif ((X[i].count("S") == 1)) and (X[i].count("Iris-versicolor") == 1):
            slCI[0][1] += 1
        elif ((X[i].count("S") == 1)) and (X[i].count("IIris-virginica") == 1):
            slCI[0][2] += 1
    elif X[i].count("M") == 1:
        sl[1] += 1
        if ((X[i].count("M") == 1)) and (X[i].count("Iris-setosa") == 1):
            slCI[0][0] += 1  
        elif ((X[i].count("M") == 1)) and (X[i].count("Iris-versicolor") == 1):
            slCI[0][1] += 1
        elif ((X[i].count("M") == 1)) and (X[i].count("IIris-virginica") == 1):
            slCI[0][2] += 1
    elif X[i].count("L") == 1:
        sl[2] += 1
        if ((X[i].count("L") == 1)) and (X[i].count("Iris-setosa") == 1):
            slCI[0][0] += 1  
        elif ((X[i].count("L") == 1)) and (X[i].count("Iris-versicolor") == 1):
            slCI[0][1] += 1
        elif ((X[i].count("L") == 1)) and (X[i].count("IIris-virginica") == 1):
            slCI[0][2] += 1

    if X[i].count("low") == 1:
        pl[0] += 1
        if ((X[i].count("S") == 1)) and (X[i].count("No") == 1):
            plCI[0][0] += 1  # class no
        else:
            plCI[0][1] += 1  # class yes
    elif X[i].count("medium") == 1:
        pl[1] += 1
        if ((X[i].count("medium") == 1)) and (X[i].count("No") == 1):
            plCI[1][0] += 1
        else:
            plCI[1][1] += 1
    elif X[i].count("high") == 1:
        pl[2] += 1
        if ((X[i].count("high") == 1)) and (X[i].count("No") == 1):
            plCI[2][0] += 1
        else:
            plCI[2][1] += 1

    if X[i].count("s_no") == 1:
        sw[0] += 1
        if ((X[i].count("s_no") == 1)) and (X[i].count("No") == 1):
            swCI[0][0] += 1  # class no
        else:
            swCI[0][1] += 1  # class yes
    elif X[i].count("s_yes") == 1:
        sw[1] += 1
        if ((X[i].count("s_yes") == 1)) and (X[i].count("No") == 1):
            swCI[1][0] += 1
        else:
            swCI[1][1] += 1

    if X[i].count("fair") == 1:
        pw[0] += 1
        if ((X[i].count("fair") == 1)) and (X[i].count("No") == 1):
            pwCI[0][0] += 1  # class no
        else:
            pwCI[0][1] += 1  # class yes
    elif X[i].count("excellent") == 1:
        pw[1] += 1
        if ((X[i].count("excellent") == 1)) and (X[i].count("No") == 1):
            pwCI[1][0] += 1
        else:
            pwCI[1][1] += 1

    if X[i].count("Iris-setosa") == 1:
        classs[0] += 1
    elif X[i].count("Iris-versicolor") == 1:
        classs[1] += 1
    elif X[i].count("Iris-virginica") == 1:
        classs[2] += 1


# calculate information gain of dataset and attb
# info D,sl,pl,sw,pw
    info_d = calculate_entropy(classs)

    for i in range(3):
        slCI[i][3] = calculate_entropy(
            [slCI[i][0], slCI[i][1], slCI[i][2]])
        swCI[i][3] = calculate_entropy(
            [swCI[i][0], swCI[i][1], swCI[i][2]])
        plCI[i][3] = calculate_entropy(
            [plCI[i][0], plCI[i][1], plCI[i][2]])
        pwCI[i][3] = calculate_entropy(
            [pwCI[i][0], pwCI[i][1], pwCI[i][2]])

# หาค่า gain แบบไม่ใช้ และใช้ฟังก์ชัน
"""
การหาแบบไม่ใช้ฟังก์ชัน
Info_slD = ((sl[0]/14)*slCI[0][2])+((sl[1]/14)*slCI[1][2])+((sl[2]/14)*slCI[2][2])
print("InfoD sl is",Info_slD)
print("sl Ci [:],[2] is",[slCI[0][2],slCI[1][2],slCI[2][2]])
print("InfoD sl is",Info_slD)
"""
Info_slD = inforD(sl, [slCI[0][2], slCI[1][2], slCI[2][2]])
Info_plD = inforD(pl, [plCI[0][2], plCI[1][2], plCI[2][2]])
Info_swD = inforD(sw, [swCI[0][2], swCI[1][2]])
Info_pwD = inforD(pw, [pwCI[0][2], pwCI[1][2]])

# แสดงผลการทำงานรอบแรก
"""
print("sl count is", sl)
print("pl count is",pl)
print("swdent count is",sw)
print("pw rating count is",pw)
print("Buy computer count is",buy)
print("sl Info relate to class",slCI)
print("pl Info relate to class",plCI)
print("swdent Info relate to class",swCI)
print("pw rating Info relate to class",pwCI)

print("Info(D) is %5.3f" % InD)
print("Info(sl S(2,3) is %5.3f" % slCI[0][2])
print("Info(sl 31...40(4,0) is %5.3f" % slCI[1][2])
print("Info(sl >40 (3,2) is %5.3f" % slCI[2][2])

print("Info(pl low(1,3) is %5.3f" % plCI[0][2])
print("Info(pl medium(2,4) is %5.3f" % plCI[1][2])
print("Info(pl high(2,2) is %5.3f" % plCI[2][2])

print("Info(swdent No (4,3) is %5.3f" % swCI[0][2])
print("Info(swdent Yes (1,6) is %5.3f" % swCI[1][2])

print("Info(pw fair(2,6) is %5.3f" % pwCI[0][2])
print("Info(pw excellent(3,3) is %5.3f" % pwCI[1][2])
print("Info sl (D) is %5.3f" % Info_slD)
print("Info pl (D) is %5.3f" % Info_plD)
print("Info swdent (D) is %5.3f" % Info_swdentD)
print("Info pw rating (D) is %5.3f" % Info_pwD)
"""
print("\n***Gain results of all dataset***")
gainsl = InD - Info_slD
print("Gain (sl) is %5.3f" % gainsl)
gainIn = InD - Info_plD
print("Gain (pl) is %5.3f" % gainIn)
gainsw = InD - Info_swdentD
print("Gain (swdent) is %5.3f" % gainsw)
gainCre = InD - Info_pwD
print("Gain (pw rating) is %5.3f" % gainCre)

# rule of root node

Result_All = [gainsl, gainIn, gainsw, gainCre]
max_gain = max(Result_All)
pos = np.argmax(Result_All)
print("max gain of attb is %5.3f" % max_gain, "position is", pos)

# วน loop แยก dataset ตาม attb sl
X2L = []  # ข้อมูลสำหรับสร้าง level 2 ที่ sl S
X2M = []  # ข้อมูลสำหรับสร้าง level 2 ที่ sl M
X2R = []  # ข้อมูลสำหรับสร้าง level 2 ที่ sl L
f1 = open("buycomL2left.txt", "w")
f2 = open("buycomL2middle.txt", "w")
f3 = open("buycomL2right.txt", "w")

for i in range(0, 15):
    if X[i].count("S") == 1:
        f1.write(str(X[i]))

    elif X[i].count("M") == 1:
        f2.write(str(X[i]))

    elif X[i].count("L") == 1:
        f3.write(str(X[i]))

# dataset of layer 2 of dtree generate
f1 = open("buycomL2left.txt", "r")
f2 = open("buycomL2middle.txt", "r")
f3 = open("buycomL2right.txt", "r")
X2L = f1.readlines()
X2M = f2.readlines()
X2R = f3.readlines()


# recursive line 14 สำหรับการสร้าง tree ชั้นที่ 2 สำหรับ dataset slS


# วน loop เพื่อนับข้อมูล แยกตามรายละเอียด attb และ class
# data ที่ sl S


# calculate information gain of dataset and attb
# info D,sl,pl,sw,pw


# หาค่า gain แบบใช้ฟังก์ชัน


# แสดงผลการทำงาน รอบ2 ฝั่งซ้าย


# recursive line 14 สำหรับการสร้าง tree ชั้นที่ 2 สำหรับ dataset sl>40


# วน loop เพื่อนับข้อมูล แยกตามรายละเอียด attb และ class
# data ที่ sl >40


# calculate information gain of dataset and attb
# info D,sl,pl,sw,pw


# หาค่า gain แบบใช้ฟังก์ชัน


# แสดงผลการทำงานรอบสอง ฝั่งขวา


# สร้าง tree
# rule extraction
# model evaluation
