def normalizeNegPos(inputvalue, min, max):
    return ( (inputvalue - min) / (max - min) - 0.5 ) * 2


def deNormalizeNegPos(normalizedValue, min, max):
    return (normalizedValue / 2 + 0.5) * (max - min) + min


def normalizePos(inputvalue, min, max):
    return (inputvalue - min) / (max - min)


def deNormalizePos(normalizedValue, min, max):
    return normalizedValue * (max - min) + min


