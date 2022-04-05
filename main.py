from os import listdir

imgs_path = "./data/images/"
xmls_path = "./data/annotations/"
filenames = listdir(imgs_path)

# train(filenames, imgs_path, xmls_path)

# from visualizers import visualize_random_image
# visualize_random_image(filenames, imgs_path, xmls_path)

# from predictions import predict_random_image
# predict_random_image(filenames, imgs_path, xmls_path)
