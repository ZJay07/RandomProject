import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import json

SEED = 42

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task2.ensemble_elm import MyEnsembleELM

def visualize_model_predictions(model_path, config_path=None, config=None, pooling = False, save_path="result.png", num_images=36):
    """
    viz method for ensemble model with annotations
    """
    # Force CPU for consistency
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Create a dataloader without shuffling to maintain order
    test_loader = DataLoader(test_dataset, batch_size=num_images, shuffle=False)

    if not config:
        # Load ensemble configuration
        config = torch.load(config_path, map_location=device)
    
    # Create ensemble model
    ensemble_model = MyEnsembleELM(
        seed=config.get('seed', 42),
        n_models=config['n_models'],
        num_feature_maps=config['num_feature_maps'],
        std_dev=config['std_dev'],
        kernel_size=config['kernel_size'],
        pooling=config.get('pooling', pooling),
    )
    
    # Load model weights
    for i in range(config['n_models']):
        model_file = os.path.join(model_path, f"model_{i}.pth")
        # Ensure all weights are loaded to CPU
        state_dict = torch.load(model_file, map_location=device)
        ensemble_model.models[i].load_state_dict(state_dict)
    
    # Make sure all models are in eval mode
    for model in ensemble_model.models:
        model.eval()
    
    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Make predictions manually to avoid any device issues
    with torch.no_grad():
        ensemble_output = torch.zeros((images.size(0), 10))
        for model in ensemble_model.models:
            outputs = model(images)
            ensemble_output += torch.nn.functional.softmax(outputs, dim=1)
        
        ensemble_output /= len(ensemble_model.models)
        _, predictions = torch.max(ensemble_output, 1)
    
    # Create individual annotated images
    annotated_images = []
    
    for i in range(num_images):
        # Get image
        img = images[i]
        
        # Denormalize
        img = img * 0.5 + 0.5
        
        # Convert to numpy format for PIL
        img_np = (np.transpose(img.numpy(), (1, 2, 0)) * 255).astype(np.uint8)
        
        # Create PIL image
        pil_img = Image.fromarray(img_np)
        
        # Resize for better visibility of annotations (optional)
        pil_img = pil_img.resize((64, 64), Image.BILINEAR)
        
        # Create a new image with white border for annotation
        width, height = pil_img.size
        annotated_img = Image.new('RGB', (width, height + 20), color='white')
        annotated_img.paste(pil_img, (0, 0))
        
        # Add draw capability
        draw = ImageDraw.Draw(annotated_img)
        
        # Get labels
        true_label = class_names[labels[i]]
        pred_label = class_names[predictions[i]]
        is_correct = labels[i] == predictions[i]
        
        # Set text color based on correctness
        text_color = 'green' if is_correct else 'red'
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except IOError:
            font = ImageFont.load_default()
        
        # Add text
        text = f"T:{true_label} P:{pred_label}"
        draw.text((2, height + 2), text, fill=text_color, font=font)
        
        # Add to list
        annotated_images.append(torch.tensor(np.transpose(np.array(annotated_img) / 255.0, (2, 0, 1))))
    
    # Stack the images
    annotated_images = torch.stack(annotated_images)
    
    # Create a grid
    grid_size = int(np.sqrt(num_images))
    grid = make_grid(annotated_images, nrow=grid_size, padding=2)
    
    # Convert to numpy and adjust for PIL
    grid_np = (np.transpose(grid.numpy(), (1, 2, 0)) * 255).astype(np.uint8)
    
    # Create final image
    result_img = Image.fromarray(grid_np)
    
    # Add title
    draw = ImageDraw.Draw(result_img)
    try:
        title_font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        title_font = ImageFont.load_default()
    
    # Add a title at the bottom
    width, height = result_img.size
    new_img = Image.new('RGB', (width, height + 40), color='white')
    new_img.paste(result_img, (0, 0))
    result_img = new_img

    # Add the title with proper positioning
    draw = ImageDraw.Draw(result_img)
    title = "Best Model (Ensemble ELM) - T: True Label, P: Predicted Label"
    draw.text((10, height + 10), title, fill='black', font=title_font)

    
    # Save the image, create path if does not exist
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    result_img.save(save_path)
    print(f"Visualization saved to {save_path}")
    
    # Calculate accuracy on these samples
    correct = (predictions == labels).sum().item()
    accuracy = 100 * correct / num_images
    print(f"Accuracy on these {num_images} samples: {accuracy:.2f}%")


if __name__ == "__main__":
    # Paths to model files
    best_regularisation_model_dir = "./task2/models/ensemble_model"
    best_regularisation_config_file = "./task2/models/ensemble_config.pth"
    
    # viz for best regularisation method
    #visualize_model_predictions(best_regularisation_model_dir, best_regularisation_config_file, save_path="./task2/montage_result/result.png")

    # viz for random search with fit_elm_ls
    # visualize_model_predictions
    best_fit_elm_ls_model = "./task2/best_hyperparameter_model"
    config_path = os.path.join("./task2/", "best_hyperparameter_config.json")

    # Load the configuration as a Python dictionary
    with open(config_path, 'r') as f:
        config = json.load(f)
    visualize_model_predictions(best_fit_elm_ls_model, config=config, save_path="./task2/montage_result/new_result.png")