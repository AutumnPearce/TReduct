import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split as sklearn_train_test_split

class ZtfData:
    IMG_SHAPE = [63,63]
    IMG_PIXELS = IMG_SHAPE[0] * IMG_SHAPE[1]
    NUM_IMGS = 11556

    def __init__(self, data = None, labels = None, zero_is=None):
        self.data, self.labels = self.get_ztf_data(zero_is)
        self.dimensions = {'one':self.IMG_PIXELS,'two':self.IMG_PIXELS,'diff':self.IMG_PIXELS}

        self.reduced_data = None

    def get_ztf_data(self,zero_is = None):
        df = pd.read_csv('candidates.csv')
        triplets = np.load('triplets.norm.npy', mmap_mode='r')

        ind = np.arange(self.NUM_IMGS)
        data = {'one':np.array(triplets[ind,:,:,0]).reshape(self.NUM_IMGS, self.IMG_PIXELS),\
                'two':np.array(triplets[ind,:,:,1]).reshape(self.NUM_IMGS, self.IMG_PIXELS),\
                'diff':np.array(triplets[ind,:,:,2]).reshape(self.NUM_IMGS, self.IMG_PIXELS)}
        labels = df.loc[ind, "label"]

        if zero_is is not None:
            labels[labels==0] = zero_is
        return data,labels

    def display_transient(self, ind = None):
        if ind is None:
            ind = np.random.randint(0, high=self.NUM_IMGS)
            
        fig = plt.figure(figsize=(8, 2), dpi=100)
        print(f'image {ind}, label {self.labels[ind]}')

        ax1 = fig.add_subplot(131)
        ax1.axis('off')
        ax1.set_title('one')
        ax1.imshow(self.data['one'][ind].reshape(self.IMG_SHAPE), origin='lower', cmap=plt.cm.bone)
        
        ax2 = fig.add_subplot(132)
        ax2.axis('off')
        ax2.set_title('two')
        ax2.imshow(self.data['two'][ind].reshape(self.IMG_SHAPE), origin='lower', cmap=plt.cm.bone)
        
        ax3 = fig.add_subplot(133)
        ax3.axis('off')
        ax3.set_title('difference')
        ax3.imshow(self.data['diff'][ind].reshape(self.IMG_SHAPE), origin='lower', cmap=plt.cm.bone)
        
        plt.show()

    def reduce(self, type='tsne', num_dimensions=2, images=['one','two','diff']):
        if type == 'tsne':  
            if num_dimensions < 4:
                redu = TSNE(n_components=num_dimensions)
            else:
                redu = TSNE(n_components=num_dimensions, method='exact')

        elif type == 'tsvd':
            redu = TruncatedSVD(n_components=num_dimensions)
        
        if self.reduced_data is None:
            self.reduced_data = self.data

        for img in images:
            self.reduced_data[img] = redu.fit_transform(self.reduced_data[img])
            self.dimensions[img] = num_dimensions

    def normalize(self, min=0, max=1, images = ['one','two','diff']):
        for img in images:
            self.reduced_data[img] = minmax_scale(self.reduced_data[img], feature_range=(min,max))

    def normalize_each(self, images = ['one','two','diff']):
        from sklearn.preprocessing import normalize as sklearn_normalize
        for img in images:
            self.reduced_data[img] = sklearn_normalize(self.reduced_data[img])

    def angle_encode(self):
        minmax_scale(self.reduced_data, feature_range=(0,math.pi))
        return

    def plot_2d_data(self):
        fig = plt.figure(figsize=(9, 3), dpi=100)

        ax1 = fig.add_subplot(131)
        ax1.set_title('one')
        ax1.scatter(self.reduced_data['one'].T[0], self.reduced_data['one'].T[1], c = self.labels, s=.5, cmap='bwr', alpha=.1)
        
        ax2 = fig.add_subplot(132)
        ax2.set_title('two')
        ax2.scatter(self.reduced_data['two'].T[0], self.reduced_data['two'].T[1], c = self.labels, s=.5, cmap='bwr', alpha=.1)
        
        ax3 = fig.add_subplot(133)
        ax3.set_title('difference')
        ax3.scatter(self.reduced_data['diff'].T[0], self.reduced_data['diff'].T[1], c = self.labels, s=.5, cmap='bwr', alpha=.1)
        
        plt.show()

    def train_test_split(self, test_size=.2, random_state=0):
        simplified_data = self._get_simplified_data()
        x_train, x_test, y_train, y_test = sklearn_train_test_split(simplified_data, self.labels.values, test_size=test_size, random_state=random_state)
        return x_train, x_test, y_train, y_test
    
    def _get_simplified_data(self, images=['one','two','diff']):
        total_dimensions = self.dimensions['one'] + self.dimensions['two'] + self.dimensions['diff']
        simplified_data = np.empty((total_dimensions, self.NUM_IMGS))

        counter = 0
        for img in images:
            for i in range(self.dimensions[img]):
                simplified_data[counter] = self.reduced_data[img].T[i]
                counter += 1
        return simplified_data.T

