import math
import numpy as np


def MSE(img1, img2):
    return np.mean(np.square(img1 - img2))


def PSNR(Max, MSE):
    return 10*math.log10(Max**2/MSE)
