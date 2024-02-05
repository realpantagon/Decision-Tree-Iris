from scipy import linalg
import numpy as np
from numpy import loadtxt
import math
from Dtreefunc import *

sl_mapping = {'sepal_length_L': 0, 'sepal_length_M': 1, 'sepal_length_H': 2}
sw_mapping = {'sepal_width_L': 0, 'sepal_width_M': 1, 'sepal_width_H': 2}
pl_mapping = {'petal_length_L': 0, 'petal_length_M': 1, 'petal_length_H': 2}
pw_mapping = {'petal_width_L': 0, 'petal_width_M': 1, 'petal_width_H': 2}

def calculate_information_gain_for_iris():
    f = open("preprocessed_iris.txt", "r")
    X = f.readlines()


    # Initialize counters and matrices
    n = 3  # Number of classes
    m = 4  # Number of attributes

    sepal_length = np.zeros(3)
    slCI = [[0 for _ in range(m)] for _ in range(n)]

    sepal_width = np.zeros(3)
    swCI = [[0 for _ in range(m)] for _ in range(n)]

    petal_length = np.zeros(3)
    plCI = [[0 for _ in range(m)] for _ in range(n)]

    petal_width = np.zeros(3)
    pwCI = [[0 for _ in range(m)] for _ in range(n)]

    iris = np.zeros(3)

    for i in range(0, len(X)):
        data = X[i].split(" ")

        sepal_length_type = sl_mapping.get(data[0])
        sepal_length[sepal_length_type] += 1

        sepal_width_type = sw_mapping.get(data[1])
        sepal_width[sepal_width_type] += 1

        petal_length_type = pl_mapping.get(data[2])
        petal_length[petal_length_type] += 1

        petal_width_type = pw_mapping.get(data[3])
        petal_width[petal_width_type] += 1

        iris_mapping = {'Iris-setosa\n': 0, 'Iris-versicolor\n': 1, 'Iris-virginica\n': 2}
        iris_type = iris_mapping.get(data[4])
        iris[iris_type] += 1

        slCI[sepal_length_type][iris_type] += 1
        swCI[sepal_width_type][iris_type] += 1
        plCI[petal_length_type][iris_type] += 1
        pwCI[petal_width_type][iris_type] += 1

    info_d = calculate_entropy(iris)

    for i in range(3):
        slCI[i][3] = calculate_entropy(
            [slCI[i][0], slCI[i][1], slCI[i][2]])
        swCI[i][3] = calculate_entropy(
            [swCI[i][0], swCI[i][1], swCI[i][2]])
        plCI[i][3] = calculate_entropy(
            [plCI[i][0], plCI[i][1], plCI[i][2]])
        pwCI[i][3] = calculate_entropy(
            [pwCI[i][0], pwCI[i][1], pwCI[i][2]])

    Info_sl_D = inforD(sepal_length, [slCI[0][3],slCI[1][3], slCI[2][3]])
    Info_pl_D = inforD(sepal_width, [swCI[0][3], swCI[1][3], swCI[2][3]])
    Info_sw_D = inforD(petal_length, [plCI[0][3], plCI[1][3], plCI[2][3]])
    Info_pw_D = inforD(petal_width, [pwCI[0][3], pwCI[1][3], pwCI[2][3]])



    # แสดงผลการทำงานรอบแรก
    # print("sl count is",sl)
    # print("pl count is",pl)
    # print("sw count is",sw)
    # print("pw count is",pw)
    # print("Iris count is",classs)

    print("sl Info relate to class",slCI)
    print("pl Info relate to class",plCI)
    print("sw Info relate to class",swCI)
    print("pw Info relate to class",pwCI)

    print("Info(D) is %5.3f" % info_d)
    print("Info(sl (< mean - SD) is %5.3f" % slCI[0][3])
    print("Info(sl (mean+-2sd) is %5.3f" % slCI[1][3])
    print("Info(sl (> mean + SD) is %5.3f" % slCI[2][3])

    print("Info(sw (< mean - SD) is %5.3f" % swCI[0][3])
    print("Info(sw (mean+-2sd) is %5.3f" % swCI[1][3])
    print("Info(sw (> mean + SD) is %5.3f" % swCI[2][3])

    print("Info(pl (< mean - SD) is %5.3f" % plCI[0][3])
    print("Info(pl (mean+-2sd) is %5.3f" % plCI[1][3])
    print("Info(pl (> mean + SD) is %5.3f" % plCI[2][3])

    print("Info(pw (< mean - SD) is %5.3f" % pwCI[0][3])
    print("Info(pw (mean+-2sd) is %5.3f" % pwCI[1][3])
    print("Info(pw (> mean + SD) is %5.3f" % pwCI[2][3])

    print("Info sl (D) is %5.3f" % Info_sl_D)
    print("Info pl (D) is %5.3f" % Info_pl_D)
    print("Info sw (D) is %5.3f" % Info_sw_D)
    print("Info pw (D) is %5.3f" % Info_pw_D)

    print("\n***Gain results of all dataset***")
    gain_sl=info_d-Info_sl_D
    print("Gain (age) is %5.3f"% gain_sl)
    gain_sw=info_d-Info_pl_D
    print("Gain (Income) is %5.3f"% gain_sw)
    gain_pl=info_d-Info_sw_D
    print("Gain (Student) is %5.3f"% gain_pl)
    gain_pw=info_d-Info_pw_D
    print("Gain (Credit rating) is %5.3f"% gain_pw)

    #rule of root node

    Result_All=[gain_sl, gain_sw, gain_pl, gain_pw]
    max_gain=max(Result_All)
    pos=np.argmax(Result_All)
    print("max gain of attb is %5.3f" % max_gain, "position is", pos)

calculate_information_gain_for_iris()