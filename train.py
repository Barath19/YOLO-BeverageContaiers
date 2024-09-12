import os 
import yaml
from ultralytics import YOLO, settings

settings.update({'clearml': False,'dvc': False, 'hub': False, 'mlflow': True, 
                 'neptune': False, 'raytune': False, 'tensorboard': False,
                'wandb': False, 'vscode_msg': False})


def train_yolov8():
    # Check if params.yaml exists using try-except
    params_file = "params.yaml"
    try:
        with open(params_file, "r") as file:
            params = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: {params_file} not found! Please provide a valid params.yaml file.")
        return

    # Extract parameters from the YAML file
    epochs = params.get("epochs", 50)
    batch_size = params.get("batch_size", 16)
    learning_rate = params.get("learning_rate", 0.01)
    model_path = params.get("model", "yolov8n.pt")
    data_path = params.get("data", "data.yaml")
    img_size = params.get("img_size", 640)

    # Load YOLOv8 model
    model = YOLO(model_path)

    # Train the model
    results = model.train(
        data=data_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        lr0=learning_rate,  # Initial learning rate
    )

    # Check if the output directory exists, if not, create it
    output_dir = "weights"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory {output_dir} created.")

    # Save the best model weights
    weights_path = os.path.join(output_dir, "best.pt")
    model.save(weights_path)

    results = model.val()

    print(f"Training complete. Model weights saved to {weights_path}")

if __name__ == "__main__":


    train_yolov8()
