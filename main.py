from engine import train
from visualizers import visualize_random_image
from prediction import predict_random_image

if __name__ == "__main__":
    # Training
    train()

    # Visualization
    visualize_random_image()
    visualize_image_by_index(533)
    
    # Prediction
    predict_random_image(
        model_path="./models/model_2022-04-15 11 34 02.128306.pth", num_preds=10
    )





