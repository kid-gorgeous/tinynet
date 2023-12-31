import keras as k 
import numpy as np
import pandas as pd 
import cv2 as cv2
from keras.models import Sequential

from termcolor import colored as c

class data:
    def __init__(self, path):
        self.path = path

    def load(self):
        cap = cv2.VideoCapture(self.path)
        while(cap.isOpened()):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow('frame',gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        


if __name__ == '__main__':
    print(c("Loading datasets", 'green'))
    wd = '/Users/evan/tinynet/datasets/videos/day_01.mp4'
    data = data(wd)
    print(c('Path: {}'.format(data.path), 'green'))

    data.load()

    # Load the datasets
    # Load the model
    # Train the model
    # Save the model
    # Test the model
    # Save the results
    # Plot the results
    # Compare the results
    