
import cv2
import sys
import random
import numpy as np 
import matplotlib.pyplot as plt

from csv import reader
from matplotlib import animation
from time import gmtime, strftime


# read csv file as a list of lists
with open('data/models/a3c_SpaceInvaders-v4.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj,delimiter=';')
    # Pass reader object to list() to get a list of lists
    list_of_rows = list(csv_reader)

    fig, axs = plt.subplots(2)
    fig.suptitle('Resultados')

    labels = list(set([row[0] for row in list_of_rows]))

    for label in labels:
        xs = [float(row[1]) for row in list_of_rows if row[0] == label]
        ys = [float(row[4]) for row in list_of_rows if row[0] == label]
        axs[0].plot(xs, ys, label=label)

    plt.xticks(np.arange(0,4000,400))
    axs[0].legend()

    for label in labels:
        xs = [float(row[1]) for row in list_of_rows if row[0] == label]
        ys = [float(row[2]) for row in list_of_rows if row[0] == label]
        axs[1].plot(xs, ys, label=label)

    plt.xticks(np.arange(0,4000,400))
    axs[1].legend()
    plt.show()
