from scipy import linalg
import numpy as np
from numpy import loadtxt
import math
from Dtreefunc import *

def irisfirststgain():
    with open("preprocessed_iris.csv", "r") as f:
        X = f.readlines()

    N = 3  # Number of classes
    M = 4  # Number of attributes

    sl = np.zeros(N)
    slCI = np.zeros((N, M))

    sw = np.zeros(N)
    swCI = np.zeros((N, M))

    pl = np.zeros(N)
    plCI = np.zeros((N, M))

    pw = np.zeros(N)
    pwCI = np.zeros((N, M))

    classs = np.zeros(N)
    # Loop through the data to count occurrences
    for i in range(0, 150):

        # Sepal Length (sl)
        print(f"Processing line {i}: {X[i]}")
        if X[i].count("sl_s") == 1:
            sl[0] += 1
            slCI[0][0] += 1 if "Iris-setosa" in X[i] else 0
            slCI[0][1] += 1 if "Iris-versicolor" in X[i] else 0
            slCI[0][2] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("sl_m") == 1:
            sl[1] += 1
            slCI[1][0] += 1 if "Iris-setosa" in X[i] else 0
            slCI[1][1] += 1 if "Iris-versicolor" in X[i] else 0
            slCI[1][2] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("sl_l") == 1:
            sl[2] += 1
            slCI[2][0] += 1 if "Iris-setosa" in X[i] else 0
            slCI[2][1] += 1 if "Iris-versicolor" in X[i] else 0
            slCI[2][2] += 1 if "Iris-virginica" in X[i] else 0

        # Sepal Width (sw)
        if X[i].count("sw_s") == 1:
            sw[0] += 1
            swCI[0][0] += 1 if "Iris-setosa" in X[i] else 0
            swCI[0][1] += 1 if "Iris-versicolor" in X[i] else 0
            swCI[0][2] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("sw_m") == 1:
            sw[1] += 1
            swCI[1][0] += 1 if "Iris-setosa" in X[i] else 0
            swCI[1][1] += 1 if "Iris-versicolor" in X[i] else 0
            swCI[1][2] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("sw_l") == 1:
            sw[2] += 1
            swCI[2][0] += 1 if "Iris-setosa" in X[i] else 0
            swCI[2][1] += 1 if "Iris-versicolor" in X[i] else 0
            swCI[2][2] += 1 if "Iris-virginica" in X[i] else 0

        # Petal Length (pl)
        if X[i].count("pl_s") == 1:
            pl[0] += 1
            plCI[0][0] += 1 if "Iris-setosa" in X[i] else 0
            plCI[0][1] += 1 if "Iris-versicolor" in X[i] else 0
            plCI[0][2] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("pl_m") == 1:
            pl[1] += 1
            plCI[1][0] += 1 if "Iris-setosa" in X[i] else 0
            plCI[1][1] += 1 if "Iris-versicolor" in X[i] else 0
            plCI[1][2] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("pl_l") == 1:
            pl[2] += 1
            plCI[2][0] += 1 if "Iris-setosa" in X[i] else 0
            plCI[2][1] += 1 if "Iris-versicolor" in X[i] else 0
            plCI[2][2] += 1 if "Iris-virginica" in X[i] else 0

        # Petal Width (pw)
        if X[i].count("pw_s") == 1:
            pw[0] += 1
            pwCI[0][0] += 1 if "Iris-setosa" in X[i] else 0
            pwCI[0][1] += 1 if "Iris-versicolor" in X[i] else 0
            pwCI[0][2] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("pw_m") == 1:
            pw[1] += 1
            pwCI[1][0] += 1 if "Iris-setosa" in X[i] else 0
            pwCI[1][1] += 1 if "Iris-versicolor" in X[i] else 0
            pwCI[1][2] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("pw_l") == 1:
            pw[2] += 1
            pwCI[2][0] += 1 if "Iris-setosa" in X[i] else 0
            pwCI[2][1] += 1 if "Iris-versicolor" in X[i] else 0
            pwCI[2][2] += 1 if "Iris-virginica" in X[i] else 0

        # Class of iris (classs)
        if "Iris-setosa" in X[i]:
            classs[0] += 1
        elif "Iris-versicolor" in X[i]:
            classs[1] += 1
        elif "Iris-virginica" in X[i]:
            classs[2] += 1

            
    # calculate information gain of dataset and attb
    # info D,sl,pl,sw,pw
    info_d = calculate_entropy([classs[0], classs[1], classs[2]])
    print("Iris-setosa",classs[0], "  Iris-versicolor",classs[1],"  Iris-virginica", classs[2])
    print("Entropy of 1st Dataset = " , info_d)

    for i in range(3):
        slCI[i][3] = calculate_entropy([slCI[i][0], slCI[i][1], slCI[i][2]])
        swCI[i][3] = calculate_entropy([swCI[i][0], swCI[i][1], swCI[i][2]])
        plCI[i][3] = calculate_entropy([plCI[i][0], plCI[i][1], plCI[i][2]])
        pwCI[i][3] = calculate_entropy([pwCI[i][0], pwCI[i][1], pwCI[i][2]])

    # Calculate information gain for each attribute
    Info_sl_D = inforD(sl, [slCI[0][3], slCI[1][3], slCI[2][3]])
    Info_sw_D = inforD(sw, [swCI[0][3], swCI[1][3], swCI[2][3]])
    Info_pl_D = inforD(pl, [plCI[0][3], plCI[1][3], plCI[2][3]])
    Info_pw_D = inforD(pw, [pwCI[0][3], pwCI[1][3], pwCI[2][3]])

    # Calculate information gain for each attribute

    print()
    print("sl count is",sl)
    print("sw count is",sw)
    print("pl count is",pl)
    print("pw count is",pw)
    print("Iris count is",classs)
    print()
    print("sl Info relate to class",slCI)
    print("pl Info relate to class",plCI)
    print("sw Info relate to class",swCI)
    print("pw Info relate to class",pwCI)
    print()
    print("Info(D) is %5.3f" % info_d)
    print("Info(sl (< mean - SD) is %5.3f" % slCI[0][3])
    print("Info(sl (mean+-2sd) is %5.3f" % slCI[1][3])
    print("Info(sl (> mean + SD) is %5.3f" % slCI[2][3])
    print()
    print("Info(sw (< mean - SD) is %5.3f" % swCI[0][3])
    print("Info(sw (mean+-2sd) is %5.3f" % swCI[1][3])
    print("Info(sw (> mean + SD) is %5.3f" % swCI[2][3])
    print()
    print("Info(pl (< mean - SD) is %5.3f" % plCI[0][3])
    print("Info(pl (mean+-2sd) is %5.3f" % plCI[1][3])
    print("Info(pl (> mean + SD) is %5.3f" % plCI[2][3])
    print()
    print("Info(pw (< mean - SD) is %5.3f" % pwCI[0][3])
    print("Info(pw (mean+-2sd) is %5.3f" % pwCI[1][3])
    print("Info(pw (> mean + SD) is %5.3f" % pwCI[2][3])
    print()
    print("Info sl (D) is %5.3f" % Info_sl_D)
    print("Info pl (D) is %5.3f" % Info_pl_D)
    print("Info sw (D) is %5.3f" % Info_sw_D)
    print("Info pw (D) is %5.3f" % Info_pw_D)
    print()
    print("\n***Gain results of all dataset***")
    gain_sl=info_d-Info_sl_D
    print("Gain of sepal length is %5.3f"% gain_sl)
    gain_sw=info_d-Info_pl_D
    print("Gain of sepal width is %5.3f"% gain_sw)
    gain_pl=info_d-Info_sw_D
    print("Gain of petal length is %5.3f"% gain_pl)
    gain_pw=info_d-Info_pw_D
    print("Gain of petal width is %5.3f"% gain_pw)

    #rule of root node

    Result_All=[gain_sl, gain_sw, gain_pl, gain_pw]
    print(Result_All)
    max_gain=max(Result_All)
    pos=np.argmax(Result_All)
    print("\nmax gain of attb is %5.3f" % max_gain, "position is", pos)


def irissecondgain():
    with open("preprocessed_iris.csv", "r") as f:
        X = f.readlines()

    N = 3  # Number of classes
    M = 3  # Number of attributes

    sl = np.zeros(N)
    slCI = np.zeros((N, M))

    pl = np.zeros(N)
    plCI = np.zeros((N, M))

    pw = np.zeros(N)
    pwCI = np.zeros((N, M))

    classs = np.zeros(2)
    # Loop through the data to count occurrences
    for i in range(0, 150):

        # Sepal Length (sl)
        print(f"Processing line {i}: {X[i]}")
        if X[i].count("sl_s") == 1:
            sl[0] += 1
            slCI[0][0] += 1 if "Iris-versicolor" in X[i] else 0
            slCI[0][1] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("sl_m") == 1:
            sl[1] += 1
            slCI[1][0] += 1 if "Iris-versicolor" in X[i] else 0
            slCI[1][1] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("sl_l") == 1:
            sl[2] += 1
            slCI[2][0] += 1 if "Iris-versicolor" in X[i] else 0
            slCI[2][1] += 1 if "Iris-virginica" in X[i] else 0


        # Petal Length (pl)
        if X[i].count("pl_s") == 1:
            pl[0] += 1
            plCI[0][0] += 1 if "Iris-versicolor" in X[i] else 0
            plCI[0][1] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("pl_m") == 1:
            pl[1] += 1
            plCI[1][0] += 1 if "Iris-versicolor" in X[i] else 0
            plCI[1][1] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("pl_l") == 1:
            pl[2] += 1
            plCI[2][0] += 1 if "Iris-versicolor" in X[i] else 0
            plCI[2][1] += 1 if "Iris-virginica" in X[i] else 0

        # Petal Width (pw)
        if X[i].count("pw_s") == 1:
            pw[0] += 1
            pwCI[0][0] += 1 if "Iris-versicolor" in X[i] else 0
            pwCI[0][1] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("pw_m") == 1:
            pw[1] += 1
            pwCI[1][0] += 1 if "Iris-versicolor" in X[i] else 0
            pwCI[1][1] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("pw_l") == 1:
            pw[2] += 1
            pwCI[2][0] += 1 if "Iris-versicolor" in X[i] else 0
            pwCI[2][1] += 1 if "Iris-virginica" in X[i] else 0

        # Class of iris (classs)
        
        if "Iris-versicolor" in X[i]:
            classs[0] += 1
        elif "Iris-virginica" in X[i]:
            classs[1] += 1

            
    # calculate information gain of dataset and attb
    # info D,sl,pl,sw,pw
    info_d = calculate_entropy([classs[0], classs[1]])
    print("  Iris-versicolor",classs[0],"  Iris-virginica", classs[1])
    print("Entropy of 1st Dataset = " , info_d)

    for i in range(3):
        slCI[i][2] = calculate_entropy([slCI[i][0], slCI[i][1]])
        plCI[i][2] = calculate_entropy([plCI[i][0], plCI[i][1]])
        pwCI[i][2] = calculate_entropy([pwCI[i][0], pwCI[i][1]])

    # Calculate information gain for each attribute
    Info_sl_D = inforD(sl, [slCI[0][2], slCI[1][2], slCI[2][2]])
    Info_pl_D = inforD(pl, [plCI[0][2], plCI[1][2], plCI[2][2]])
    Info_pw_D = inforD(pw, [pwCI[0][2], pwCI[1][2], pwCI[2][2]])

    # Calculate information gain for each attribute

    print()
    print("sl count is",sl)
    print("pl count is",pl)
    print("pw count is",pw)
    print("Iris count is",classs)
    print()
    print("sl Info relate to class",slCI)
    print("pl Info relate to class",plCI)
    print("pw Info relate to class",pwCI)
    print()
    print("Info(D) is %5.3f" % info_d)
    print("Info(sl (< mean - SD) is %5.3f" % slCI[0][2])
    print("Info(sl (mean+-2sd) is %5.3f" % slCI[1][2])
    print("Info(sl (> mean + SD) is %5.3f" % slCI[2][2])
    print()
    print("Info(pl (< mean - SD) is %5.3f" % plCI[0][2])
    print("Info(pl (mean+-2sd) is %5.3f" % plCI[1][2])
    print("Info(pl (> mean + SD) is %5.3f" % plCI[2][2])
    print()
    print("Info(pw (< mean - SD) is %5.3f" % pwCI[0][2])
    print("Info(pw (mean+-2sd) is %5.3f" % pwCI[1][2])
    print("Info(pw (> mean + SD) is %5.3f" % pwCI[2][2])
    print()
    print("Info sl (D) is %5.3f" % Info_sl_D)
    print("Info pl (D) is %5.3f" % Info_pl_D)
    print("Info pw (D) is %5.3f" % Info_pw_D)
    print()
    print("\n***Gain results of all dataset***")
    gain_sl=info_d-Info_sl_D
    print("Gain of sepal length is %5.3f"% gain_sl)
    gain_sw=info_d-Info_pl_D
    print("Gain of sepal width is %5.3f"% gain_sw)
    gain_pw=info_d-Info_pw_D
    print("Gain of petal width is %5.3f"% gain_pw)

    #rule of root node

    Result_All=[gain_sl, gain_sw, gain_pw]
    print(Result_All)
    max_gain=max(Result_All)
    pos=np.argmax(Result_All)
    print("\nmax gain of attb is %5.3f" % max_gain, "position is", pos)


def iristhirdgain():
    with open("preprocessed_iris.csv", "r") as f:
        X = f.readlines()

    N = 3  # Number of classes
    M = 3  # Number of attributes

    sl = np.zeros(N)
    slCI = np.zeros((N, M))

    sw = np.zeros(N)
    swCI = np.zeros((N, M))

    classs = np.zeros(2)
    # Loop through the data to count occurrences
    for i in range(0, 150):

        # Sepal Length (sl)
        print(f"Processing line {i}: {X[i]}")
        if X[i].count("sl_s") == 1:
            sl[0] += 1
            slCI[0][0] += 1 if "Iris-versicolor" in X[i] else 0
            slCI[0][1] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("sl_m") == 1:
            sl[1] += 1
            slCI[1][0] += 1 if "Iris-versicolor" in X[i] else 0
            slCI[1][1] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("sl_l") == 1:
            sl[2] += 1
            slCI[2][0] += 1 if "Iris-versicolor" in X[i] else 0
            slCI[2][1] += 1 if "Iris-virginica" in X[i] else 0

        # Sepal width (sw)
        if X[i].count("sw_s") == 1:
            sw[0] += 1
            swCI[0][0] += 1 if "Iris-versicolor" in X[i] else 0
            swCI[0][1] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("sw_m") == 1:
            sw[1] += 1
            swCI[1][0] += 1 if "Iris-versicolor" in X[i] else 0
            swCI[1][1] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("sw_l") == 1:
            sw[2] += 1
            swCI[2][0] += 1 if "Iris-versicolor" in X[i] else 0
            swCI[2][1] += 1 if "Iris-virginica" in X[i] else 0  

        # Class of iris (classs)
        if "Iris-versicolor" in X[i]:
            classs[0] += 1
        elif "Iris-virginica" in X[i]:
            classs[1] += 1

            
    # calculate information gain of dataset and attb
    # info D,sl,pl,sw,pw
    info_d = calculate_entropy([classs[0], classs[1]])
    print("Iris-versicolor",classs[0],"  Iris-virginica", classs[1])
    print("Entropy of 1st Dataset = " , info_d)

    for i in range(3):
        slCI[i][2] = calculate_entropy([slCI[i][0], slCI[i][1]])
        swCI[i][2] = calculate_entropy([swCI[i][0], swCI[i][1]])

    # Calculate information gain for each attribute
    Info_sl_D = inforD(sl, [slCI[0][2], slCI[1][2], slCI[2][2]])
    Info_sw_D = inforD(sw, [swCI[0][2], swCI[1][2], swCI[2][2]])

    # Calculate information gain for each attribute

    print()
    print("sl count is",sl)
    print("sw count is",sw)
    print("Iris count is",classs)
    print()
    print("sl Info relate to class",slCI)
    print("sw Info relate to class",swCI)
    print()
    print("Info(D) is %5.3f" % info_d)
    print("Info(sl (< mean - SD) is %5.3f" % slCI[0][2])
    print("Info(sl (mean+-2sd) is %5.3f" % slCI[1][2])
    print("Info(sl (> mean + SD) is %5.3f" % slCI[2][2])
    print()
    print("Info(sw (< mean - SD) is %5.3f" % swCI[0][2])
    print("Info(sw (mean+-2sd) is %5.3f" % swCI[1][2])
    print("Info(sw (> mean + SD) is %5.3f" % swCI[2][2])
    print()
    print("Info sl (D) is %5.3f" % Info_sl_D)
    print("Info sw (D) is %5.3f" % Info_sw_D)
    print()
    print("\n***Gain results of all dataset***")
    gain_sl=info_d-Info_sl_D
    print("Gain of sepal length is %5.3f"% gain_sl)
    gain_sw=info_d-Info_sw_D
    print("Gain of sepal width is %5.3f"% gain_sw)

    #rule of root node

    Result_All=[gain_sl, gain_sw]
    print(Result_All)
    max_gain=max(Result_All)
    pos=np.argmax(Result_All)
    print("\nmax gain of attb is %5.3f" % max_gain, "position is", pos)


def irisfourthgain():
    with open("preprocessed_iris.csv", "r") as f:
        X = f.readlines()

    N = 3  # Number of classes
    M = 3  # Number of attributes

    sl = np.zeros(N)
    slCI = np.zeros((N, M))

    classs = np.zeros(2)
    # Loop through the data to count occurrences
    for i in range(0, 150):

        # Sepal Length (sl)
        print(f"Processing line {i}: {X[i]}")
        if X[i].count("sl_s") == 1:
            sl[0] += 1
            slCI[0][0] += 1 if "Iris-versicolor" in X[i] else 0
            slCI[0][1] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("sl_m") == 1:
            sl[1] += 1
            slCI[1][0] += 1 if "Iris-versicolor" in X[i] else 0
            slCI[1][1] += 1 if "Iris-virginica" in X[i] else 0
        elif X[i].count("sl_l") == 1:
            sl[2] += 1
            slCI[2][0] += 1 if "Iris-versicolor" in X[i] else 0
            slCI[2][1] += 1 if "Iris-virginica" in X[i] else 0


        # Class of iris (classs)
        if "Iris-versicolor" in X[i]:
            classs[0] += 1
        elif "Iris-virginica" in X[i]:
            classs[1] += 1
      
    # calculate information gain of dataset and attb
    # info D,sl,pl,sw,pw
    info_d = calculate_entropy([classs[0], classs[1]])
    print("Iris-versicolor",classs[0],"  Iris-virginica", classs[1])
    print("Entropy of 1st Dataset = " , info_d)

    for i in range(3):
        slCI[i][2] = calculate_entropy([slCI[i][0], slCI[i][1]])

    # Calculate information gain for each attribute
    Info_sl_D = inforD(sl, [slCI[0][2], slCI[1][2], slCI[2][2]])

    # Calculate information gain for each attribute

    print()
    print("sl count is",sl)
    print("Iris count is",classs)
    print()
    print("sl Info relate to class",slCI)
    print()
    print("Info(D) is %5.3f" % info_d)
    print("Info(sl (< mean - SD) is %5.3f" % slCI[0][2])
    print("Info(sl (mean+-2sd) is %5.3f" % slCI[1][2])
    print("Info(sl (> mean + SD) is %5.3f" % slCI[2][2])
    print()
    print("Info sl (D) is %5.3f" % Info_sl_D)
    print()
    print("\n***Gain results of all dataset***")
    gain_sl=info_d-Info_sl_D
    print("Gain of sepal length is %5.3f"% gain_sl)

    #rule of root node

    Result_All=[gain_sl]
    print(Result_All)
    max_gain=max(Result_All)
    pos=np.argmax(Result_All)
    print("\nmax gain of attb is %5.3f" % max_gain, "position is", pos)