"""
download_photos.py
By Melanie Appleby
Written for Project 5 at Metis

Downloads each image from given shopstyle url and create dictionary with metadata.
"""

import os
import urllib
from pymongo import MongoClient
import pickle

client = MongoClient('localhost', 27017)
db = client.shopstyle_database
products_collection = db.products_by_category

categories = ['dresses','womens-tops']
product_dict = {}

i = 1
for cat in categories:
	for product in products_collection.find({'category':str(cat)}).batch_size(50):
		image_name = 'image' + str(i) + '.jpg'
		urllib.urlretrieve(product['image_url'],'products3/'+str(cat)+'/'+image_name)
		i+=1

		product_dict[image_name] = [product['retailer'], product['brandedname'],
									product['price'], product['retailer_url']]

with open('product_dict.pickle', 'wb') as handle:
	pickle.dump(product_dict, handle)
