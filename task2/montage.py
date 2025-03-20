import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image, ImageDraw, ImageFont
import os
import sys

SEED = 42

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from task2.ensemble_elm import MyEnsembleELM


def visualise_model_predictions(
    model_path,
    config_path=None,
    config=None,
    pooling=False,
    save_path="result.png",
    num_images=36,
    device="cpu",
):
    """
    Create a visualisation grid showing model predictions on test images.

    This function loads an ensemble model, runs predictions on a sample of test images,
    and creates an annotated grid showing the true and predicted labels for each image.
    The grid is saved as an image file.

    Args:
        model_path (str): Path to the directory containing model weight files
        config_path (str, optional): Path to the model configuration file. Required if config is None with a default value of `None`
        config (dict, optional): Dictionary containing model configuration. Required if config_path is None with a default value of `None`
        pooling (bool, optional): Whether to use pooling in the model with a default value of `False`
        save_path (str, optional): Path to save the visualisation image with a default value of "result.png"
        num_images (int, optional): Number of images to include in the visualisation with a default value of `36`
        device (str, optional): Device to train on ('cpu' or 'cuda') with a default value of "cpu"

    Returns:
        None: The function saves the visualisation to disk but does not return any value
              It also prints the accuracy on the visualised samples.
    """

    # default is cpu, can be changed to cuda if available
    device = torch.device(device)
    print(f"Using device: {device}")

    # CIFAR-10 class names
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # compact short codes for better visualisation
    class_codes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    # Load test dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Create a dataloader without shuffling to maintain order
    test_loader = DataLoader(test_dataset, batch_size=num_images, shuffle=False)

    if not config:
        # Load ensemble configuration
        config = torch.load(config_path, map_location=device, weights_only=True)

    # Create ensemble model with specified configs
    ensemble_model = MyEnsembleELM(
        seed=config.get("seed", 42),
        n_models=config["n_models"],
        num_feature_maps=config["num_feature_maps"],
        std_dev=config["std_dev"],
        kernel_size=config["kernel_size"],
        pooling=config.get("pooling", pooling),
    )

    # Load model weights
    for i in range(config["n_models"]):
        model_file = os.path.join(model_path, f"model_{i}.pth")
        # Ensure all weights are loaded to device
        state_dict = torch.load(model_file, map_location=device, weights_only=True)
        ensemble_model.models[i].load_state_dict(state_dict)

    # make all models eval
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

    # Batch annotation
    annotated_images = []

    for i in range(num_images):

        # Get image and process to format for PIL
        img = images[i]
        img = img * 0.5 + 0.5
        img_np = (np.transpose(img.numpy(), (1, 2, 0)) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        pil_img = pil_img.resize((64, 64), Image.BILINEAR)

        # Create white baorder and annotate
        width, height = pil_img.size
        annotated_img = Image.new("RGB", (width, height + 15), color="white")
        annotated_img.paste(pil_img, (0, 0))

        draw = ImageDraw.Draw(annotated_img)

        # Get labels
        true_label_idx = labels[i].item()
        pred_label_idx = predictions[i].item()
        is_correct = true_label_idx == pred_label_idx

        # Set green for correct and red for wrong
        text_color = "green" if is_correct else "red"

        # loading font, if available
        try:
            font = ImageFont.truetype("arial.ttf", 10)
        except IOError:
            font = ImageFont.load_default()

        # Add class number text
        text = f"T:{class_codes[true_label_idx]} P:{class_codes[pred_label_idx]}"

        draw.text((2, height + 2), text, fill=text_color, font=font)

        annotated_images.append(
            torch.tensor(np.transpose(np.array(annotated_img) / 255.0, (2, 0, 1)))
        )
    annotated_images = torch.stack(annotated_images)

    # Create a grid
    grid_size = int(np.sqrt(num_images))
    grid = make_grid(annotated_images, nrow=grid_size, padding=2)
    grid_np = (np.transpose(grid.numpy(), (1, 2, 0)) * 255).astype(np.uint8)

    # Final image creation
    result_img = Image.fromarray(grid_np)

    # Add title and legend
    draw = ImageDraw.Draw(result_img)
    try:
        title_font = ImageFont.truetype("arial.ttf", 16)
        legend_font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        title_font = ImageFont.load_default()
        legend_font = ImageFont.load_default()

    width, height = result_img.size
    legend_height = 150
    new_img = Image.new("RGB", (width, height + legend_height), color="white")
    new_img.paste(result_img, (0, 0))
    result_img = new_img

    # Add the title
    draw = ImageDraw.Draw(result_img)
    title = "Best Model (Ensemble ELM) - T: True Label, P: Predicted Label"
    draw.text((10, height + 10), title, fill="black", font=title_font)

    # Add legend
    legend_y = height + 40
    legend_x = 10
    draw.text((legend_x, legend_y), "Class codes:", fill="black", font=legend_font)

    # Create a clearer legend with full class names
    col_width = width // 2
    items_per_col = 5
    for i in range(len(class_names)):
        col = i // items_per_col
        row = i % items_per_col
        x = legend_x + col * col_width
        y = legend_y + 20 + row * 16
        legend_text = f"{i}: {class_names[i]}"
        draw.text((x, y), legend_text, fill="black", font=legend_font)

    # Save the image, create path if does not exist
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    result_img.save(save_path)
    print(f"Visualisation saved to {save_path}")

    # Calculate accuracy on these samples
    correct = (predictions == labels).sum().item()
    accuracy = 100 * correct / num_images
    print(f"Accuracy on these {num_images} samples: {accuracy:.2f}%")

    # visualise each image with true and predicted labels on the terminal
    pred_list = predictions.cpu().numpy().tolist()
    label_list = labels.cpu().numpy().tolist()

    # Print in a grid format to match the visualisation
    for row in range(grid_size):
        row_labels = []
        row_preds = []

        for col in range(grid_size):
            idx = row * grid_size + col
            true_label = class_names[label_list[idx]]
            pred_label = class_names[pred_list[idx]]

            # Add to the row lists
            row_labels.append(f"{true_label:10}")
            row_preds.append(f"{pred_label:10}")

        # Print the rows
        print(f"True labels (row {row + 1}):  " + "  ".join(row_labels))
        print(f"Predictions (row {row + 1}):  " + "  ".join(row_preds))
        print("")
