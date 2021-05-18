import cv2
import numpy as np
from tqdm import tqdm

def get_hog_feature(img, winsize=(32, 32), blocksize=(8, 8), blockstride=(4, 4), cellsize=(4, 4), bin=9):
    hog = cv2.HOGDescriptor(winsize, blocksize, blockstride, cellsize, bin)
    feature = hog.compute(img)
    return feature

def load_data(pth):
    f = open('{}/label.txt'.format(pth), 'r')
    X = list()
    Y = list()
    for line in tqdm(f.readlines()):
        file, label = line.strip().split()[:2]
        label = int(label)
        if label > 9:
            label = 0
        img = cv2.imread('{}/{}'.format(pth, file))
        X.append(get_hog_feature(img))
        Y.append(label)
    f.close()
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def label2onehot(label):
    return np.array([[1 if j == label[i] else 0 for j in range(10)] for i in range(label.shape[0])])

if __name__ == '__main__':
    img = cv2.imread('./data/processed/train/3.png')
    cv2.imshow('test', img)
    cv2.waitKey(0)
    f = get_hog_feature(img)
    print(f.shape)
    print(f)
    print(np.sum(f))
