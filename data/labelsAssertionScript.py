import pandas as pd
import numpy as np
from glob import glob 
from matplotlib import pyplot as plt
import pdb

def loadData(path,identifier="*food*csv",header=0):
    fName = glob(path+identifier)
    if not fName:
        print("Datafile not found in given directory!")
        return False
    if len(fName) > 1:
        print("Datafile not uniquily identified!")
        return False
    else:
        df = pd.read_csv(fName[0],encoding='Windows-1252', header=header)
        return df

path = "./"
df = loadData(path)

labels = loadData(path, identifier="labels.txt", header=None)

k = 0
FEATURE = "Energy"
for lab in labels.iterrows():
    iName_1 = ''
    iName_2 = ''
    iName_3 = ''
    itemName = iName = lab[1].values[0]
    
    # if find in first doesn't find space its just the word
    if iName.find(" ") == -1:
        iName_1 = iName
    # else cut out the first word and pass the rest on
    else:
        idx = iName.find(" ")
        iName_1 = iName[0:idx]
        iName = iName[idx+1:]
        # if on the rest find doesnt find space it is the just the word
        if iName.find(" ") == -1:
            iName_2 = iName
        # else cut out the first word and the rest is the third word
        else:
            idx = iName.find(" ")
            iName_2 = iName[0:idx]
            iName_3 = iName[idx+1:]
            
    if len(iName_1)>0 & len(iName_2)>0 & len(iName_3)>0:
        subSet = df[df['Name'].str.contains(iName_1) |
                    df['Name'].str.contains(iName_2) |
                    df['Name'].str.contains(iName_3)]
    if len(iName_1)>0 & len(iName_2)>0:
        subSet = df[df['Name'].str.contains(iName_1) |
                    df['Name'].str.contains(iName_2)]
    if len(iName_1)>0:
        subSet = df[df['Name'].str.contains(iName_1)]
    # if not subSet.empty:
        # k = k+1
        # plt.figure()
        # plt.boxplot(subSet[FEATURE])
        # if k == 9:
            # plt.show(block=False)
        # else:
            # plt.show(block=True)
    # else:
    if subSet.empty:
        print(itemName)