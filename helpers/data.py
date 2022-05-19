from os import listdir
from os.path import isfile, join
import numpy as np
import cv2

PATH = './data/images'


def get_data():
    data = np.array([cv2.cvtColor(cv2.imread(join(PATH, f)), cv2.COLOR_BGR2RGB)
                     for f in listdir(PATH) if isfile(join(PATH, f))])
    return data
