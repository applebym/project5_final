"""
image_search_app.py
By Melanie Appleby
Written for Project 5 at Metis

Contains python code for Flask app.
"""

import os
from flask import Flask, request, render_template, send_from_directory, jsonify
import fashion_image_search
import pickle

# Global variables
PATH_TO_IMAGE_FOLDER = '/Users/MelanieAppleby/ds/metis/metisgh/project5_final/upload_file_python/src/static/products'
NUM_CLUSTERS = 30
NUM_IMAGES_RETURNED = 32

# Load product meta data (retailer, product name, price, url)
with open('product_dict.pickle', 'rb') as handle:
    products_dict = pickle.load(handle)


app = Flask(__name__, static_folder='static')
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/get_similar", methods=["POST"])
def get_similar():
    data = request.json
    image_name = data['name']
    path_to_image = os.path.join('static/images', image_name)
    
    # Use tensorflow to get category of clothing
    category, _ = fashion_image_search.run_inference_on_image(path_to_image)
    category = fashion_image_search.reformat(category)
    
    # Use opencv to find keypoints
    keypoints = fashion_image_search.find_keypoints(path_to_image)
    
    # Using pre-trained k-means cluster model, assign labels to query photo's keypoints
    kmeans = fashion_image_search.load_pickle(category + '_kmeans')
    labels = fashion_image_search.get_labels(keypoints, kmeans)
    
    # Create histogram of query photo's keypoints
    hist = fashion_image_search.get_histogram(labels)
    
    # Using pre-trained nearest-neighbors model, find 'k' nearest neighbors
    knn = fashion_image_search.load_pickle(category + '_knn')
    dist, ind = fashion_image_search.get_similar_images(hist, knn, NUM_IMAGES_RETURNED * 5)
    
    # Get list of stock images within predicted category
    images = os.listdir(PATH_TO_IMAGE_FOLDER + '/' + category)
    images = [x for x in images if x != '.DS_Store']
    images = sorted(images)

    # Get names of similar clothing items, not including duplicate items
    image_list = []
    dists_included = []
    count = 0
    i = 0
    while count < NUM_IMAGES_RETURNED:
        if dist[i] in dists_included:
            i += 1
        else:
            image_list.append(images[ind[i]])
            dists_included.append(dist[i])
            count += 1
            i += 1
    
    # Get metadata on similar clothing items
    retailers = []
    names = []
    prices = []
    urls = []
    for image in image_list:
        info = products_dict[image]
        retailers.append(info[0])
        names.append(info[1])
        prices.append(info[2])
        urls.append(info[3])

    results = {"images": image_list, "category": category, "retailers": retailers,
               "names": names, "prices": prices, "urls": urls}
    return jsonify(results)


@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'static/images')
    
    # Create folder for query image
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    
    # Save uploaded query photo to this directory
    for upload in request.files.getlist("file"):
        filename = upload.filename
        destination = "/".join([target, filename])
        upload.save(destination)

    return render_template("upload.html", image_name=filename)


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("static/images", filename)


if __name__ == "__main__":
    app.run(port=4555, debug=True)
