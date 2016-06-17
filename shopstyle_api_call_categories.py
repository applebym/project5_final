import shopstyle
import json
from pymongo import MongoClient
from tqdm import tqdm

client = MongoClient('localhost', 27017)
db = client.shopstyle_database
# category_collection = db.products_by_category
category_collection = db.products_by_category_more

api = shopstyle.ShopStyle(api_key="uid4900-34204930-88")

cdata = api.categories()['categories']
categories = []
for category in cdata:
        categories.append((category['id'], category['parentId']))

def specify(categories):
        children = [x[0] for x in categories]
        parents = [x[1] for x in categories]
        categories_specific = []
        for childId, parentId in categories:
                if parentId == 'womens-clothes' or parentId == 'mens-clothes':
                        categories_specific.append(childId)
        return categories_specific

specific = specify(categories)
specific = [x for x in specific if x not in ['womens-accessories','bridal','maternity-clothes',
'petites','plus-sizes','teen-girls-clothes','mens-accessories','mens-big-and-tall',
'teen-guys-clothes']]

products = {}
for category in specific:
        print category
        for i in tqdm(range(2000, 5000, 50)):
                pdata = api.search(cat = category.lower(), offset = i, limit = 50)
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
