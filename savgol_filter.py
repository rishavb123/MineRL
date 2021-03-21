import numpy as np

def savgol_filter(y, window_size, degree): #Savitzky-Golay Filter
    if len(y) == 1:
        return y
    y_smooth = []
    for x in range(len(y)):
        a = max(int(x - window_size / 2), 0)
        b = min(int(x + window_size / 2), len(y))
        p_vec = np.polyfit(range(a, b), y[a:b], degree)
        p = lambda x: np.dot([x ** d for d in range(degree + 1)], np.flip(p_vec))
        y_smooth.append(p(x))

    return y_smooth