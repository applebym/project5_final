import os
import urllib
from pymongo import MongoClient

categories = ['mens-sweaters', 'mens-sweatshirts', 'mens-swimsuits', 'mens-ties', 'mens-underwear-and-socks', 'mens-watches-and-jewelry']

client = MongoClient('localhost', 27017)
db = client.shopstyle_database
products_collection = db.products_by_category

i = 48000
for cat in categories:
    # if not(os.path.isdir('products2/'+str(cat))):
    os.makedirs('products2/'+str(cat))
    for product in products_collection.find({'category':str(cat)}).batch_size(50):
            urllib.urlretrieve(product['image_url'], 
            	'products2/'+str(cat)+'/image'+str(i)+'.jpg')
            i+=1
