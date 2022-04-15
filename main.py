# from engine import train

# train()

# from visualizers import visualize_random_image

# visualize_random_image()
# visualize_image_by_index(533)

from prediction import predict_random_image

predict_random_image(
    model_path="./models/model_2022-04-15 11 34 02.128306.pth", num_preds=10
)
