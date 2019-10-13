import pandas as pd
import numpy as np
from glob import glob 
import pdb

def loadData(path,identifier="*food*csv"):
    fName = glob(path+identifier)
    if not fName:
        print("Datafile not found in given directory!")
        return False
    if len(fName) > 1:
        print("Datafile not uniquily identified!")
        return False
    else:
        df = pd.read_csv(fName[0],encoding='Windows-1252')
        return df

path = "./"

df = loadData(path)
df.head()   