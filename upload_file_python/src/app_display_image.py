import os
from uuid import uuid4
from flask import Flask, request, render_template, send_from_directory, jsonify
import fashion_image_search
import pickle


### Load Pickles ###
with open('product_dict.pickle', 'rb') as handle:
    products_dict = pickle.load(handle)

# kmeans = fashion_image_search.load_pickle('dresses_kmeans')
# knn = fashion_image_search.load_pickle('dresses_knn')


modelFullPath = '/Users/MelanieAppleby/ds/metis/metisgh/project5_final/upload_file_python/src/output_graph.pb'
labelsFullPath = '/Users/MelanieAppleby/ds/metis/metisgh/project5_final/upload_file_python/src/output_labels.txt'
path_to_image_folder = '/Users/MelanieAppleby/ds/metis/metisgh/project5_final/upload_file_python/src/static/products'
NUM_CLUSTERS = 30
NUM_IMAGES_RETURNED = 30

app = Flask(__name__, static_folder='static')
# app = Flask(__name__, static_folder="images")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/get_similar", methods=["POST"])
def get_similar():
    data = request.json
    print 'Recieved data'
    image_name = data['name']
    path_to_image = os.path.join('static/images', image_name)
    category, _ = fashion_image_search.run_inference_on_image(path_to_image)
    print category
    category = fashion_image_search.reformat(category)
    keypoints = fashion_image_search.find_keypoints(path_to_image)
    kmeans = fashion_image_search.load_pickle('dresses_kmeans')
    labels = fashion_image_search.get_labels(keypoints, kmeans)
    hist = fashion_image_search.get_histogram(labels)
    knn = fashion_image_search.load_pickle('dresses_knn')
    dist, ind = fashion_image_search.get_similar_images(hist, knn, NUM_IMAGES_RETURNED * 4)
    images = os.listdir(path_to_image_folder + '/' + category)
    images = [x for x in images if x != '.DS_Store']
    images = sorted(images)
    
    for i in range(len(dist)):
        print images[ind[i]], dist[i]

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
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)

    return render_template("upload.html", image_name=filename)


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("static/images", filename)


if __name__ == "__main__":
    app.run(port=4555, debug=True)
