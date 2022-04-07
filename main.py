from engine import train

train()

# from visualizers import visualize_random_image

# visualize_random_image()
# visualize_image_by_index(533)

from prediction import predict_random_image

predict_random_image("./models/model_2022-04-07 22:13:42.649693.pth")
