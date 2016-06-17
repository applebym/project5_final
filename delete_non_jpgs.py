import imghdr
import os

path = '/Users/MelanieAppleby/ds/metis/metisgh/project5_final/products2'
sub_folders = os.listdir(path)[1:]
for folder in sub_folders:
	folder_path = path + '/' + folder
	files = os.listdir(folder_path)[1:]
	for file in files:
		img_type = imghdr.what(folder_path + '/' + file)
		if img_type != 'jpeg':
			print folder, file, img_type

