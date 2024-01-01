import keras as k 
import numpy as np
import pandas as pd 
import cv2 as cv2
from keras.models import Sequential

from termcolor import colored as c

class data:
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename

    def load(self):
        cap = cv2.VideoCapture(self.path)
        
        while(cap.isOpened()):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('{}'.format(self.filename),frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        


if __name__ == '__main__':
    print(c("Loading datasets", 'green'))
    filename = 'day_01.mp4'
    wd = '/Users/evan/tinynet/datasets/videos/{}'.format(filename)
    data = data(wd, filename)
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
    