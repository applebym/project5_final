import numpy as np
import tensorflow as tf
import sys
import cv2
import pandas as pd
import pickle
import os

modelFullPath = '/Users/MelanieAppleby/ds/metis/metisgh/project5_final/output_graph.pb'
labelsFullPath = '/Users/MelanieAppleby/ds/metis/metisgh/project5_final/output_labels.txt'
NUM_CLUSTERS = 100

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
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
        # for node_id in top_k:
        #     human_string = labels[node_id]
        #     score = predictions[node_id]
        #     print('%s (score = %.5f)' % (human_string, score))

        answer1 = labels[top_k[0]]
        answer2 = labels[top_k[1]]

        if predictions[top_k[0]] / predictions[top_k[1]] > 3:
        	return answer1, None
        else:
        	return answer1, answer2
	

def reformat(cat):
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


def main():
	path_to_image = sys.argv[1]
	
	## Classify image ##
	cat1, cat2 = run_inference_on_image(path_to_image)
	cat1 = reformat(cat1)
	print 'This is a ', cat1
	# if cat2 != None:
	# 	cat2 = reformat(cat2)

	## Return similar photos ##
	path_to_image_folder = '/Users/MelanieAppleby/ds/metis/metisgh/project5_final/products_test'
	keypoints = find_keypoints(path_to_image)
	
	# Category 1 #
	kmeans = load_pickle(cat1 + '_kmeans')
	labels = get_labels(keypoints, kmeans)
	hist = get_histogram(labels)
	knn = load_pickle(cat1 + '_knn')
	dist, ind = get_similar_images(hist, knn, k = 10)
	images = os.listdir(path_to_image_folder + '/' + cat1)[1:]
	# top_10_dists = zip(range(10), dist[:10])
	# print 'Image, Distance'
	# for i in range(10):
	# 	print images[ind[i]], dist[i]

	# Category 2 # 
	# if cat2 != None:
	# 	kmeans2 = load_pickle(cat2 + '_kmeans')
	# 	labels2 = get_labels(keypoints, kmeans2)
	# 	hist2 = get_histogram(labels2)
	# 	knn2 = load_pickle(cat2 + '_knn')
	# 	dist2, ind2 = get_similar_images(hist2, knn2, k = 10)
	# 	images2 = os.listdir(path_to_image_folder + '/' + cat2)[1:]
	# 	top_10_dists2 = zip(range(10,20), dist2[:10])
	# 	# print 'Image, Distance'
	# 	# for i in range(10):
	# 	# 	print images2[ind2[i]], dist2[i]
	# 	combined_dists = top_10_dists + top_10_dists2
	# 	sorted_dists = sorted(combined_dists, key=lambda x: x[1])
	# 	print 'Image, Distance'
	# 	for i in range(10):
	# 		index = sorted_dists[i][0]
	# 		if index <= 9:
	# 			print images[ind[index]], dist[index]
	# 		else:
	# 			print images2[ind[index - 10]], dist2[index - 10]
	# else:
	print 'Image, Distance'
	for i in range(10):
		print images[ind[i]], dist[i]

if __name__ == '__main__':
    main()