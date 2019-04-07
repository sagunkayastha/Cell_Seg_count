import tensorflow as tf
import numpy as np
from keras.models import model_from_json
import cv2
import sys
import argparse


class Cell_Seg_Count:

    def __init__(self, file):
        self.seg_model = self.load_mdl()[0]
        self.count_model = self.load_mdl()[1]
        self.load_wgt()
        self.image = file
        self.index = 0

    def load_mdl(self):
        
        json_file = open('models/seg_model.json', 'r')
        lj1 = json_file.read()
        json_file.close()
        seg_model = model_from_json(lj1)
        
        json_file = open('models/count_model.json', 'r')
        lj2 = json_file.read()
        json_file.close()
        count_model = model_from_json(lj2)

        return [seg_model, count_model]

    def load_wgt(self):
        self.seg_model.load_weights('weights/seg_weight.h5')
        self.count_model.load_weights('weights/count_weight.h5')

    def Segmentation(self):      
        X = self.image
        X = np.expand_dims(X,0)
        A = self.seg_model.predict(X)    
        
        return A

    def Counting(self):
        # y = np.load('../NN_count/train_mask.npy')

        self.seg_out = self.Segmentation()
        out = self.count_model.predict(self.seg_out)
        
        print(f'out - {out}')

    def count_all(self):
        X = np.load('../bb_cell/train_mask.npy')
        A = self.seg_model.predict(X[:600]) 
        y = np.load('../NN_count/train_count.npy')

        out = self.count_model.predict(A)
        for i in range(len(out)):
            print(f'out - {out[i]}, Label - {y[i]}')

    def plot(self):
        cv2.imshow('Input Image', self.image)
        cv2.imshow('Segmentation', self.seg_out[0])
        cv2.waitKey(0)
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', type=str, default='test_images/C14.TIF',
                        help='test_image')
    parser.add_argument('--A', type=str, default='None',
                        help='count all')
    args = parser.parse_args()
    
    if args.A == 'All':
        a = Cell_Seg_Count(None)
        a.count_all()
        
    else:

        img = cv2.imread(args.t)[:,:,0]
        # img = cv2.imread('test_images/C57.TIF')[:,:,0]
        img = cv2.resize(img, (256,256))
        img = np.expand_dims(img,-1)

        a = Cell_Seg_Count(img)
        a.Counting()
        a.plot()


if __name__ == '__main__':
    main()