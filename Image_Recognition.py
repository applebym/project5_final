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
from sklearn.cluster import KMeans

NUM_CLUSTERS = 20

def create_keypoints(path_to_images):
    """
    Computes SIFT keypoints for all images
    """
    keypoints_df = pd.DataFrame()
    keypoints_index = []

    images = os.listdir(path_to_images)[1:]  # the first object is a .DS_STORE file
    create_pickle(name = 'images', obj = images)

    sift = cv2.SIFT()
    for image in tqdm(images):
        img = cv2.imread(path_to_images + '/' + image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        df = pd.DataFrame(des)
        keypoints_df = keypoints_df.append(df, ignore_index=True)
        keypoints_index.append(len(df))
    return keypoints_df, keypoints_index


def find_keypoints(image):
    """
    Finds and returns SIFT keypoints for a single image
    """
    sift = cv2.SIFT()
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    keypoints = pd.DataFrame(des)
    return keypoints


def create_clusters(keypoints):
    """
    Computes clusters of keypoints and return the labels for each keypoint 
    and the K-Means model
    """
    kmeans = KMeans(n_clusters = NUM_CLUSTERS, n_init = 10)
    kmeans.fit(keypoints)
    create_pickle(name = 'kmeans', obj = kmeans)
    labels = kmeans.labels_
    return kmeans, labels


def get_labels(keypoints, kmeans):
    """
    Labels keypoints according to kmeans model
    """
    labels = kmeans.predict(keypoints)
    return labels


def create_histograms(labels, keypoints_index):
    """
    Creates histogram of labeled keypoints for each image
    keypoints_index is a list containing the number of keypoints in each image
    """
    hists = []
    prev = 0 
    for num in tqdm(keypoints_index):
        curr = prev + num
        hist, _ = np.histogram(labels[prev:curr], bins = NUM_CLUSTERS, 
            range = [0, NUM_CLUSTERS])
        hist = hist.astype('float')    # normalize
        hist /= hist.sum()    # normalize
        hists.append(hist)
        prev = curr
    return hists


def get_histogram(labels):
    """
    Creates histogram for a single image
    """
    hist, _ = np.histogram(labels, bins = NUM_CLUSTERS, range = [0, NUM_CLUSTERS])
    hist = hist.astype('float')    # normalize
    hist /= hist.sum()    # normalize
    return hist


def run_knn(hists):
    """
    Runs KNN model for all images
    """
    knn = NearestNeighbors()
    knn.fit(hists)
    return knn


def get_similar_images(hist, knn, k):
    """
    Return k similar images for given image
    """
    print hist
    dist, ind = knn.kneighbors(hist, n_neighbors = k)
    return dist[0], ind[0]


def create_pickle(name, obj):
    """
    Pickles the given object
    """
    with open(name + '.pickle', 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(name):
    """
    Loads the given object from its pickle and return it
    """
    with open(name + '.pickle', 'rb') as handle:
        obj = pickle.load(handle)
    return obj


def main():
    method = sys.argv[1]
    if method == 'create':
        path_to_images = sys.argv[2]
        
        print 'Computing keypoints'
        # keypoints_df, keypoints_index = create_keypoints(path_to_images)
        # create_pickle(name = 'keypoints_df', obj = keypoints_df)
        # create_pickle(name = 'keypoints_index', obj = keypoints_index)
        keypoints_df = load_pickle('keypoints_df')
        keypoints_index = load_pickle('keypoints_index')

        print 'Creating clusters'
        kmeans, labels = create_clusters(keypoints_df)
        create_pickle(name = 'labels', obj = labels)

        print 'Organizing keypoints into histograms'
        hists = create_histograms(labels, keypoints_index)
        create_pickle(name = 'hists', obj = hists)

        print 'Creating KNN model'
        knn = run_knn(hists)
        create_pickle(name = 'knn', obj = knn)

    if method == 'get':
        path_to_image = sys.argv[2]
        path_to_images = sys.argv[3]
        num_similar_images = int(sys.argv[4])

        keypoints = find_keypoints(path_to_image)
        kmeans = load_pickle('kmeans')
        labels = get_labels(keypoints, kmeans)
        hist = get_histogram(labels)
        knn = load_pickle('knn')
        dist, ind = get_similar_images(hist, knn, num_similar_images)

        images = os.listdir(path_to_images)[1:]
        print 'Image Name, Distance'
        for i in range(num_similar_images):
            print images[ind[i]], dist[i]


if __name__ == '__main__':
    main()


