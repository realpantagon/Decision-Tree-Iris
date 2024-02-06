import numpy as np


def classification():
    f = open("preprocessed_iris.csv", "r")
    x = f.readlines()
    result = []
    choice = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    true_false = np.zeros(2)
    iris_type = np.zeros(3)

    for i in range(0, len(x)):
        data = x[i].split(",")
        if data[2] == "pl_s":
            result.append(choice[0])
            iris_type[0] += 1
        elif data[2] == "pl_m":
            if data[3] == "pw_s":
                result.append(np.random.choice(choice))
            elif data[3] == "pw_m":
                if data[0] == "sl_s":
                    result.append(choice[1])
                    iris_type[1] += 1
                elif data[0] == "sl_m":
                    if data[1] == "sw_h":
                        result.append(np.random.choice(choice))
                    else:
                        result.append(choice[1])
                        iris_type[1] += 1
                else:
                    result.append(choice[1])
                    iris_type[1] += 1
            else:
                result.append(choice[2])
                iris_type[2] += 1
        else:
            result.append(choice[2])
            iris_type[2] += 1

        ans = data[4].strip('\n')
        if result[-1] == ans:
            true_false[0] += 1
        else:
            true_false[1] += 1
    print(result)
    print(iris_type)
    print(true_false)
    print(f"Accuracy value = {true_false[0]/sum(true_false)}")
