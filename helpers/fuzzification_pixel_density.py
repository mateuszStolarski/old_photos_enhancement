import numpy as np

# Gaussian Function:


def G(x, mean, std):
    return np.exp(-0.5*np.square((x-mean)/std))

# Membership Functions:


def ExtremelyDark(x, M):
    return G(x, -50, M/6)


def VeryDark(x, M):
    return G(x, 0, M/6)


def Dark(x, M):
    return G(x, M/2, M/6)


def SlightlyDark(x, M):
    return G(x, 5*M/6, M/6)


def SlightlyBright(x, M):
    return G(x, M+(255-M)/6, (255-M)/6)


def Bright(x, M):
    return G(x, M+(255-M)/2, (255-M)/6)


def VeryBright(x, M):
    return G(x, 255, (255-M)/6)


def ExtremelyBright(x, M):
    return G(x, 305, (255-M)/6)
