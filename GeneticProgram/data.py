import numpy as np


class Data(object):
    """
    Class to read data, and  manipulate the data such to be shuffled, as well as split th
    """

    def __init__(self, text_file):
        self.text_file = text_file

    def read_data(self, shuffle_d=False):
        """
        Function to load in the text file. Function splits the data into two sets.
        set 1: company data
        set 2: company data labels - either a 0 or 1.
        :return: tuple - (company data, company class)
        """
        from numpy import loadtxt
        from numpy.random import shuffle
        cfd = loadtxt(self.text_file)  # read in the data

        # if the shuffle flag true, then shuffle the data.
        if shuffle_d is True:
            shuffle(cfd)

        class_labels_cfd = cfd[:, -1]  # get the classification categories - [0,1].
        class_labels_cfd = [int(x) for x in class_labels_cfd]
        class_labels_cfd = np.asarray(class_labels_cfd, dtype=int)

        data_cfd = cfd[:, 0:-1]
        return data_cfd, class_labels_cfd

    # need to make train test split.