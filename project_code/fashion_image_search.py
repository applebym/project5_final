"""
fashion_image_search.py
By Melanie Appleby
Written for Project 5 at Metis

Contains code to perform image search on given query image.
"""

import numpy as np
import tensorflow as tf
import sys
import cv2
import pandas as pd
import pickle
import os

modelFullPath = '/Users/MelanieAppleby/ds/metis/metisgh/project5_final/upload_file_python/src/output_graph.pb'
labelsFullPath = '/Users/MelanieAppleby/ds/metis/metisgh/project5_final/upload_file_python/src/output_labels.txt'
NUM_CLUSTERS = 30

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    with tf.gfile.FastGFile(modelFullPath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(imagePath):
    answer = None

    if not tf.gfile.Exists(imagePath):
        tf.logging.fatal('File does not exist %s', imagePath)
        return answer

    image_data = tf.gfile.FastGFile(imagePath, 'rb').read()

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:

        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-5:][::-1]  # Getting top 5 predictions
        f = open(labelsFullPath, 'rb')
        lines = f.readlines()
        labels = [str(w).replace("\n", "") for w in lines]

        answer1 = labels[top_k[0]]
        answer2 = labels[top_k[1]]

        if predictions[top_k[0]] / predictions[top_k[1]] > 3:
        	return answer1, None
        else:
        	return answer1, answer2
	

def reformat(cat):
    """
    Reformats category string 
    """
	list_of_words = cat.split()
	one_word = '-'.join(list_of_words)
	return one_word


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


def get_labels(keypoints, kmeans):
    """
    Labels keypoints according to kmeans model
    """
    labels = kmeans.predict(keypoints)
    return labels


def get_histogram(labels):
    """
    Creates histogram for a single image
    """
    hist, _ = np.histogram(labels, bins = NUM_CLUSTERS, range = [0, NUM_CLUSTERS])
    hist = hist.astype('float')    # normalize
    hist /= hist.sum()    # normalize
    return hist


def get_similar_images(hist, knn, k):
    """
    Return k similar images for given image
    """
    dist, ind = knn.kneighbors(hist, n_neighbors = k)
    return dist[0], ind[0]


def load_pickle(name):
    """
    Loads the given object from its pickle and return it
    """
    with open('pickle/' + name + '.pickle', 'rb') as handle:
        obj = pickle.load(handle)
    return obj