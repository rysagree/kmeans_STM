# -*- coding: utf-8 -*-
"""
Spyder Editor

kmeans module
"""

import math
import random
import os
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import scipy.io as sio
from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd

# load function
def MATLAB_matrix_to_array(fileName):
    """
        Converts a Matlab '.mat' file into a numpy array.
        Assumes the name of the '.mat' file is the same as the name of the variable

        Parameters
        ----------
        fileName : str
            String containing the file name.

        Returns
        -------
        array
            a numpy array version of the inputted matlab matrix
    """
    
    pathAndFileName = os.getcwd()+'/dIdV_data/'+fileName
    matrix = sio.loadmat(pathAndFileName, appendmat=True)
    print(matrix.keys())
    array = matrix[fileName]
    print("Shape of array: ",np.shape(array))
    return array

# general purpose functions

def flatten(data):
    """
    Flattens array by reducing the spatial dimension. 
    (x,y,energy) -> (x*y, energy)
    or
    (x,y) -> (x*y)
        
    Parameters
    ----------
    data : array_like
        3D array in the shape (x,y,energy), or,
        2D array in the shape (x,y)

    Returns
    -------
    flat_data : array
        2D array in the shape (x*y, energy), or,
        1D array in the shape (x*y)
    """
    if np.ndim(data)== 3:
        flat_data = np.reshape(data, (np.shape(data)[0]*np.shape(data)[1], np.shape(data)[2]))
    elif np.ndim(data)== 2:
        flat_data = np.reshape(data, (np.shape(data)[0]*np.shape(data)[1]))
    else:
        print("Invalid dimensionality of data. Options are 2D or 3D.")
    return flat_data


def fold(flat_data, other_data):
    """
    Unflattens array in the spatial dimension. 
    (x*y,energy) -> (x,y, energy)
    or
    (x*y) -> (x,y)
        
    Parameters
    ----------
    flat_data : array
        2D array in the shape (x*y,energy), or,
        1D array in the shape (x*y)
    other_data : array
        Other dataset with the spatial dimensions desired

    Returns
    -------
    data : array
        3D array in the shape (x, y, energy), or,
        2D array in the shape (x, y)
    """
    if np.ndim(flat_data)== 2:
        data = np.reshape(flat_data, (np.shape(other_data)[0],np.shape(other_data)[1], np.shape(other_data)[2]))
    elif np.ndim(flat_data)== 1:
        data = np.reshape(flat_data, (np.shape(other_data)[0],np.shape(other_data)[1]))
    else:
        print("Invalid dimensionality of data. Options are 1D or 2D.")
    return data


#Define bias values
def bias(minV, maxV, data):
    """
    Defines array of bias voltage values 
        
    Parameters
    ----------
    minV : float
        value of the first voltage value.
    maxV : float
        value of the last voltage value
    data : array
        data with the energy dimension desired

    Returns
    -------
    V : array
        1D array that contains values from minV to maxV of the length of the energy dimension of data
    """
    V = np.linspace(minV, maxV, num=len(data[0][0]))
    return V

def cluster_average(data, centroid_indices, startV, endV):
    """
    Calculate average dIdV from a given cluster

    Parameters
    ----------
    data : array
        array of data in shape (x,y,energy)
    centroid_indices : int
        index corresponding to centroid
    startV : float
        starting voltage value
    endV : float
        final voltage value

    """

    data_flat=flatten(data)
    counter=0
    for i in centroid_indices:
        if counter==0:
            data_cluster=data_flat[i]
        else:
            data_cluster = np.vstack((data_cluster, data_flat[i]))
        counter+=1

    energy = bias(startV, endV, data)
    dIdV_avg = np.average(data_cluster, axis=0)
    
    return energy, dIdV_avg

# kmeans related functions

def my_kmeans(dataset, n_clusters):
    """
        Runs k-means clustering algorithm on hyperspectral data
        Uses 'k-means++' centroid initialization for quicker optimization and is seeded for reproducibility
        See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        
        Parameters
        ----------
        dataset : array_like
            data in the shape (n_samples, n_features), ie: (75625, 81)
        n_clusters : int
            number of clusters for k-means to generate

        Returns
        -------
        labels : array
            array of cluster labels for the data
        centroids : array
            array of cluster centroids
        score : float
            value of Calinski-Harabasz indices for the k-means solution. Equivalent to WCSS / BCSS
            See https://scikit-learn.org/stable/modules/clustering.html#calinski-harabasz-index
    """
    data = flatten(dataset)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++',n_init=100, max_iter=100, random_state=0).fit(data)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    score=0
    if n_clusters !=1:
        score = metrics.calinski_harabasz_score(data, labels)
        print('WCSS / BCSS = {:0.6e}'.format(score))
    print('k-means complete')
    return labels, centroids, score


def save_kmeans(fileName, labels, centroids, data):
    """
        Write and save cluster assignments to a file
        
        Parameters
        ----------
        fileName : str
            String of the base file name to save the results as.
        labels : array
            array of cluster labels
        centroids : array
            array of k-means centroids
        data : array
            data that the k-means results were generated from
    """
    img_labels = fold(labels, data)
    np.savetxt(fileName+"_labels.csv", img_labels, delimiter=',')
    np.savetxt(fileName+"_centroids.csv", centroids, delimiter=',')


def import_kmeans(fileName):
    """
        Import previously calculated kMeans labels and centroids
        
        Parameters
        ----------
        fileName : str
            String of the base file name the results were saved as.
            
        Returns
        -------
        labels : array
            array of previously calculated cluster labels
        centroids : array
            array of previously calculated cluster centroids
    """
    labels = pd.read_csv(fileName+"_labels.csv", sep = ",", header = None)
    centroids = pd.read_csv(fileName+"_centroids.csv", sep = ",", header = None)
    return labels.values, centroids.values


def kmeans_Plot(data,labels, centroids, dataset_params, dataset_dict, colour_dict):
    """
    Plots results from k-means analysis
        
    Parameters
    ----------
    data : array
        data that k-means was run on in shape (x,y,energy)
    labels : array
        array of cluster labels in shape (x,y)
    centroids : array
        array of k-means centroids
    dataset_params : dict
        dictionary of dataset fixed parameters (startV and endV)
    dataset_dict : dict
        dictionary of dataset label identities
    colour_dict : dict
        dictionary of colour values associated with label identities
    """
    
    energy = bias(dataset_params['startV'], dataset_params['endV'], data)
    number_centroids = np.shape(centroids)[0]
    colour_list = ["" for i in range(number_centroids)]
    
    #Plotting
    fig,(ax1,ax2) = plt.subplots(1,2)
    
    for key,value in dataset_dict.items():
        colour_list[value] = colour_dict[key] #creating for panel 2
        ax1.plot(energy,centroids[value],label=key, marker='',linestyle='-',lw=2, color=colour_dict[key])
    
    ax1.set_ylabel("dI/dV (arb.)")
    ax1.set_xlabel("Bias (V)")
    ax1.set_title("k-means centroids")
    ax1.legend()

    cmap = colors.ListedColormap(colour_list)
    ax2.imshow(labels, cmap=cmap)
    ax2.set_title("Map of cluster assignments")
    ax2.set_xticks([])
    ax2.set_yticks([])
    
# plotting


def sampling_subplot(ax, centroid_index, centroids, index_choice, color, startV, endV, data, n_sample):
    """
    Plot a sampling of spectra belonging to a given centroid

    Parameters
    ----------
    ax : axis
        figure axis object
    centroid_index : int
        index corresponding to centroid
    centroids : array
        array of k-means calculated centroids
    index_choice : array
        array of indices of point spectra to plot
    color : str
        colour for centroid
    startV : float
        starting voltage value
    endV : float
        final voltage value
    data : array
        array of data in shape (x,y,energy)
    """
    energy = bias(startV, endV, data)
    flattened_data = flatten(data)
    
    for i in range(n_sample):
        ax.plot(energy, flattened_data[index_choice[i]], marker='',ls='-', color='gray', alpha=0.3,lw=0.5)
    
    #plot the centroid spectrum
    ax.plot(energy,centroids[centroid_index], marker='',linestyle='-', color=color, lw=2)
    ax.set_xlim([startV, endV])
    ax.set_ylim([-0.2E-11,6.5E-11])
    #ax.yaxis.set_ticks([])
    ax.xaxis.set_ticks(( -2, -1., 0., 1.))
    return





