import numpy as np
import os
import os.path

def LoadMNIST_0Lab(file_img, file_lab, th=230):
    if not os.path.isfile(file_img):
        raise Exception('Data file {} not found. Make sure to unzip the file dat/mnist.zip'.format(file_img))
    if not os.path.isfile(file_lab):
        raise Exception('Labels file {} not found. Make sure to unzip the file dat/mnist.zip'.format(file_lab))
    x = np.load(file_img)
    y = np.load(file_lab)
    def Pad(x):
        x_padded = np.empty([x.shape[0],32,32], dtype=np.float32)
        for ix in range(x.shape[0]):
            pic = x[ix].reshape([28,28])
            pic = np.pad(pic, pad_width=2, mode='constant', constant_values=0)
            x_padded[ix] = pic.astype(dtype=np.float32)
        return x_padded
    def Binarize(x, th):
        return np.where(x > th, 1, 0).astype(dtype=np.float32)
    # get a balanced set of 0 label and non-0 label datapoints
    ixsOfLab = []
    for f in range(10):
        ixs = np.argwhere(y == f).reshape([-1])
        np.random.shuffle(ixs)
        ixsOfLab.append(ixs)
    minlen = (len(ixsOfLab[0]) // 9) * 9
    ixsOfLab[0] = ixsOfLab[0][:minlen]
    for f in range(1, 10):
        ixsOfLab[f] = ixsOfLab[f][:minlen // 9]
    ixsOfLab = np.concatenate(ixsOfLab, axis=0)
    np.random.shuffle(ixsOfLab)
    x = x[ixsOfLab]
    y = np.where(y[ixsOfLab] == 0, 0, 1).astype(dtype=np.float32)
    x = Pad(Binarize(x, th))
    return x, y

