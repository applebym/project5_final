import os
import urllib
from pymongo import MongoClient

path_to_main = '/Users/MelanieAppleby/ds/metis/metisgh/project5_final/products2'
already_done = ['.DS_Store', 'dresses','jackets','jeans','jewelry','mens-athletic',
'mens-blazers-and-sport-coats','mens-jeans','mens-outerwear','mens-pants',
'mens-shirts','mens-shorts','mens-sleepwear']
categories = [x for x in os.listdir(path_to_main) if x not in already_done]

client = MongoClient('localhost', 27017)
db = client.shopstyle_database
products_collection = db.products_by_category_more

i = 94991
for cat in categories:
	print cat
	for product in products_collection.find({'category':str(cat)}).batch_size(50):
		urllib.urlretrieve(product['image_url'],'products2/'+str(cat)+'/image'+str(i)+'.jpg')
		i+=1
