import pandas as pd
import numpy as np

# Function specified for preprocessing

def crear_data():
    np.random.seed(10)
    #  Location of Training data
    Data = pd.read_csv("E:/Python_New_Project/0801_Project/out_final 4_BACKUP.csv")

    print(Data.head(5))
    dtat = []

    label = []
    print("-------------------------------------------------------")
    for i in range(0, len(Data) - 200):
        temp = []
        for j in range(i, 200 + i):
            number = float(Data['Rsend'].values[j])
            temp.append(number)
        dtat.append(temp)
        label.append(Data['Rsend'].values[j + 1])
    dtat = np.array(dtat)
    dtat = dtat.reshape(dtat.shape[0], dtat.shape[1], 1)
    print("dtat.shape", dtat.shape)
    label = np.array(label)
    label = label.reshape(label.shape[0], 1)
    print(dtat[-1])
    print("---->", label[0:15])
    return dtat, label
