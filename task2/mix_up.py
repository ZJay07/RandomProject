import numpy as np
import torch
from torchvision.utils import make_grid
from PIL import Image


class MyMixUp:
    def __init__(self, alpha=1.0, seed=42):
        self.alpha = alpha

        self.random_state = np.random.RandomState(seed)
        torch.manual_seed(seed)

    def __call__(self, x, y, device="cpu"):
        # x: input tensor Batch of images
        # y: target tensor Batch of labels
        if device == "cuda":
            x = x.cuda()
            y = y.cuda()

        batch_size = x.size(0)
        if self.alpha > 0:
            lam = self.random_state.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # create shuffled indices
        index = torch.randperm(batch_size, device=x.device)

        # mix images: \lambda * x + (1-\lambda)·x[index]
        mixed_x = lam * x + (1 - lam) * x[index, :]

        # Return mixed images and both sets of labels with lambda
        return mixed_x, y, y[index], lam

    def visualize_mixup_grid(
        self, dataset, num_samples=16, save_path="mixup.png", seed=42
    ):
        """
        Visualize a grid of MixUp augmented images
        """
        # Set random seed for reproducibility
        np.random.seed(seed)

        # Get random samples from the dataset
        indices = np.random.choice(len(dataset), size=num_samples * 2, replace=False)

        # Prepare for visualization
        mixed_images = []
        lambdas = []
        labels1 = []
        labels2 = []

        for i in range(num_samples):
            img1, label1 = dataset[indices[i]]
            img2, label2 = dataset[indices[i + num_samples]]

            # Convert to tensor if necessary
            if not isinstance(img1, torch.Tensor):
                img1 = torch.tensor(img1)
            if not isinstance(img2, torch.Tensor):
                img2 = torch.tensor(img2)

            # Generate lambda for this pair
            if self.alpha > 0:
                # lam = 0.5 works too for fixed mixing
                lam = np.random.beta(self.alpha, self.alpha)
            else:
                lam = 1

            print(f"Lambda for mix {i}: {lam}")

            # Mix images
            mixed_img = lam * img1 + (1 - lam) * img2

            # Store information
            mixed_images.append(mixed_img)
            lambdas.append(lam)
            labels1.append(label1)
            labels2.append(label2)

        # Convert list to tensor
        mixed_images = torch.stack(mixed_images)

        # Create a grid of images
        grid = make_grid(mixed_images, nrow=4, normalize=True, padding=2)
        grid_np = (np.transpose(grid.cpu(), (1, 2, 0)).numpy() * 255).astype(np.uint8)

        # Create PIL image and save
        img = Image.fromarray(grid_np)
        img.save(save_path)

        print(f"Grid visualization saved to {save_path}")

        return mixed_images, lambdas, labels1, labels2


if __name__ == "__main__":
    from torchvision import datasets, transforms

    # Define transformation - no normalization for clearer visualization
    transform = transforms.Compose([transforms.ToTensor()])

    # Load dataset
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    # Create MixUp instance and visualize
    mixup = MyMixUp(alpha=1.0, seed=42)

    # Create grid visualization
    mixup.visualize_mixup_grid(train_dataset, num_samples=16, save_path="mixup.png")
