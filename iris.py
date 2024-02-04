from Dtreefunc import *  # Import your decision tree functions

# Load the Iris dataset
f = open("iris.txt", "r")

X = f.readlines()

# Prepare space to store calculation results separated by class
L = 3  # Number of classes
N = 4  # Number of features
M = 3  # Number of rows

sepal_length = np.zeros(3)
sepal_length_CI = [[0 for i in range(M)] for j in range(N)]

sepal_width = np.zeros(3)
sepal_width_CI = [[0 for i in range(M)] for j in range(N)]

petal_length = np.zeros(3)
petal_length_CI = [[0 for i in range(M)] for j in range(N)]

petal_width = np.zeros(3)
petal_width_CI = [[0 for i in range(M)] for j in range(N)]

iris_class = np.zeros(3)
iris_class_CI = [[0 for i in range(M)] for j in range(L)]

# Loop to count data based on features and class
for i in range(0, len(X)):
    if "Iris-setosa" in X[i]:
        iris_class[0] += 1
        if "Iris-setosa" in X[i]:
            iris_class_CI[0][0] += 1  # class setosa
        else:
            iris_class_CI[0][1] += 1  # other classes
    elif "Iris-versicolor" in X[i]:
        iris_class[1] += 1
        if "Iris-versicolor" in X[i]:
            iris_class_CI[1][0] += 1  # class versicolor
        else:
            iris_class_CI[1][1] += 1  # other classes
    elif "Iris-virginica" in X[i]:
        iris_class[2] += 1
        if "Iris-virginica" in X[i]:
            iris_class_CI[2][0] += 1  # class virginica
        else:
            iris_class_CI[2][1] += 1  # other classes

    # Additional code for sepal_length, sepal_width, petal_length, petal_width

# Calculate information gain of dataset and attributes
# info D, sepal_length, sepal_width, petal_length, petal_width
info = np.zeros(5)
InD = entropy(iris_class[0], iris_class[1], iris_class[2])

sepal_length_CI[0][2] = entropy(sepal_length_CI[0][0], sepal_length_CI[0][1])
sepal_width_CI[0][2] = entropy(sepal_width_CI[0][0], sepal_width_CI[0][1])
petal_length_CI[0][2] = entropy(petal_length_CI[0][0], petal_length_CI[0][1])
petal_width_CI[0][2] = entropy(petal_width_CI[0][0], petal_width_CI[0][1])

# Additional code for calculating information gain of other features

Info_sepal_length_D = inforD(sepal_length, [sepal_length_CI[0][2], sepal_length_CI[1][2], sepal_length_CI[2][2]])
Info_sepal_width_D = inforD(sepal_width, [sepal_width_CI[0][2], sepal_width_CI[1][2], sepal_width_CI[2][2]])
Info_petal_length_D = inforD(petal_length, [petal_length_CI[0][2], petal_length_CI[1][2], petal_length_CI[2][2]])
Info_petal_width_D = inforD(petal_width, [petal_width_CI[0][2], petal_width_CI[1][2], petal_width_CI[2][2]])

# Additional code for calculating information gain of other features

print("\n***Gain results of all dataset***")
gainSepalLength = InD - Info_sepal_length_D
print("Gain (Sepal Length) is %5.3f" % gainSepalLength)
gainSepalWidth = InD - Info_sepal_width_D
print("Gain (Sepal Width) is %5.3f" % gainSepalWidth)
gainPetalLength = InD - Info_petal_length_D
print("Gain (Petal Length) is %5.3f" % gainPetalLength)
gainPetalWidth = InD - Info_petal_width_D
print("Gain (Petal Width) is %5.3f" % gainPetalWidth)

# rule of root node

Result_All = [gainSepalLength, gainSepalWidth, gainPetalLength, gainPetalWidth]
max_gain = max(Result_All)
pos = np.argmax(Result_All)
print("max gain of attribute is %5.3f" % max_gain, "position is", pos)
