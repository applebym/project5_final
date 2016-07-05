"""
shopstyle_api_pull.py
By Melanie Appleby
Written for Project 5 at Metis

Pulls data from shopstyle API and stores in MongoDB database.
"""

import shopstyle
import json
from pymongo import MongoClient
from tqdm import tqdm

client = MongoClient('localhost', 27017)
db = client.shopstyle_database
category_collection = db.products_by_category

api = shopstyle.ShopStyle(api_key="uid4900-34204930-88")

categories = ['dresses','womens-tops']

products = {}
for category in categories:
        for i in tqdm(range(0, 15000, 50)):
                pdata = api.search(cat = category, offset = i, limit = 50)
                for i in range(len(pdata['products'])):
                        retailer = pdata['products'][i]['retailer']['name']
                        brand = pdata['products'][i]['brandedName']
                        price = pdata['products'][i]['price']
                        image_url = pdata['products'][i]['image']['sizes']['Original']['url']
                        retailer_url = pdata['products'][i]['clickUrl']
                        document = {
                                'category': category,
                                'retailer': retailer,
                                'brandedname': brand,
                                'price': price,
                                'image_url': image_url,
                                'retailer_url': retailer_url
                        }
                        category_collection.insert(document)
