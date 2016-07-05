"""
fashion_image_search_training.py
By Melanie Appleby
Written for Project 5 of Metis

Contains code that creates trained models on stock images for image search. 
"""

from pymongo import MongoClient
import urllib
import cv2
import numpy as np
import os
import sys
import pickle
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from sqlalchemy import create_engine

NUM_CLUSTERS = 20

def create_keypoints(path_to_images):
    """
    Computes SIFT keypoints for all images
    """
    keypoints_df = pd.DataFrame()
    keypoints_index = []

    images = os.listdir(path_to_images)
    images = [x for x in images if x != '.DS_Store']

    sift = cv2.SIFT()
    for image in tqdm(images):
        img = cv2.imread(path_to_images + '/' + image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        if des is None:
            os.remove(path_to_images + '/' + image)
            print 'removing image: ', image
            continue
        else:
            df = pd.DataFrame(des)
            keypoints_df = keypoints_df.append(df, ignore_index=True)
            keypoints_index.append(len(df))
    return keypoints_df, keypoints_index


def create_clusters(keypoints):
    """
    Computes clusters of keypoints and return the labels for each keypoint 
    and the K-Means model
    """
    kmeans = MiniBatchKMeans(n_clusters = NUM_CLUSTERS, n_init = 10)
    kmeans.fit(keypoints)
    labels = kmeans.labels_
    return kmeans, labels


def create_histograms(labels, keypoints_index):
    """
    Creates histogram of labeled keypoints for each image
    keypoints_index is a list containing the number of keypoints in each image
    """
    hists = []
    prev = 0 
    for num in keypoints_index:
        curr = prev + num
        hist, _ = np.histogram(labels[prev:curr], bins = NUM_CLUSTERS, 
            range = [0, NUM_CLUSTERS])
        hist = hist.astype('float')    # normalize
        hist /= hist.sum()    # normalize
        hists.append(hist)
        prev = curr
    return hists


def run_knn(hists):
    """
    Runs KNN model for all images
    """
    knn = NearestNeighbors()
    knn.fit(hists)
    return knn


def create_pickle(name, obj):
    """
    Pickles the given object
    """
    with open('pickle/' + name + '.pickle', 'wb') as handle:
        pickle.dump(obj, handle)


def main():
    path_to_main = sys.argv[1]
    sub_folders = [x for x in os.listdir(path_to_main) if x != '.DS_Store']   # First item is always .DS_STORE
    
    for folder in sub_folders:
        folder_path = path_to_main + '/' + folder
        
        # Find keypoints
        keypoints_df, keypoints_index = create_keypoints(folder_path)
        create_pickle(name = folder + '_keypoints_df', obj = keypoints_df)
        create_pickle(name = folder + '_keypoints_index', obj = keypoints_index)
        
        # Cluster keypoints 
        kmeans, labels = create_clusters(keypoints_df)
        create_pickle(name = folder + '_kmeans', obj = kmeans)
        create_pickle(name = folder + '_labels', obj = labels)

        # Create histograms
        hists = create_histograms(labels, keypoints_index)
        create_pickle(name = folder + '_hists', obj = hists)

        # Train unsupervised nearest-neighbors model
        knn = run_knn(hists)
        create_pickle(name = folder + '_knn', obj = knn)

if __name__ == '__main__':
    main()


