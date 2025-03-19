import numpy as np
import torch
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from PIL import Image


class MyMixUp:
    """
    Implements the MixUp data augmentation technique for image classification.

    MixUp creates new training samples by linearly interpolating between pairs of
    images and their corresponding labels. The interpolation coefficient is sampled
    from a Beta distribution controlled by the alpha parameter.
    """

    def __init__(self, alpha=1.0, seed=42):
        """
        Init the MixUp augmentation.

        Args:
            alpha (float, optional): Parameter for the Beta distribution used to sample
                                    the mixing coefficient. Larger values create more
                                    diverse mixtures with a default value of `1.0`
            seed (int, optional): Random seed for reproducibility with a default value of `42`
        """
        self.alpha = alpha

        self.random_state = np.random.RandomState(seed)
        torch.manual_seed(seed)

    def __call__(self, x, y, device="cpu"):
        """
        Apply MixUp augmentation to a batch of images and labels.

        Args:
            x (torch.Tensor): Batch of input images of shape (batch_size, channels, height, width)
            y (torch.Tensor): Batch of corresponding labels of shape (batch_size,)
            device (str, optional): Device to perform operations on with a default value of "cpu"

        Returns:
            tuple: A tuple containing:
                - mixed_x (torch.Tensor): Mixed images of shape (batch_size, channels, height, width)
                - targets_a (torch.Tensor): Original labels of shape (batch_size,)
                - targets_b (torch.Tensor): Labels of shuffled images of shape (batch_size,)
                - lam (float): Mixing coefficient from Beta distribution
        """
        if device == "cuda":  # default is cpu but allowed cuda for flexibility
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

    def visualise_mixup_grid(
        self, dataset, num_samples=16, save_path="mixup.png", seed=42
    ):
        """
        Visualise a grid of MixUp augmented images and save to a file.

        Args:
            dataset (torch.utils.data.Dataset): Dataset containing images to visualise
            num_samples (int, optional): Number of mixed images to generate with a default value of `16`
            save_path (str, optional): Path to save the visualisation image with a default value of "mixup.png"
            seed (int, optional): Random seed for reproducibility with a default value of `42`

        Returns:
            tuple: A tuple containing:
                - mixed_images (torch.Tensor): Tensor of mixed images
                - lambdas (list): List of lambda values used for mixing
                - labels1 (list): List of first labels in the mix
                - labels2 (list): List of second labels in the mix
        """
        # Set random seed for reproducibility
        np.random.seed(seed)

        # Get random samples from the dataset
        indices = np.random.choice(len(dataset), size=num_samples * 2, replace=False)

        # Prepare for visualisation
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

        print(f"Grid visualisation saved to {save_path}")

        return mixed_images, lambdas, labels1, labels2


if __name__ == "__main__":
    print("=== Producing Mixup Visualisation ===")

    # no normalisation for clearer visualisation
    transform = transforms.Compose([transforms.ToTensor()])

    # Load dataset
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    # Create MixUp instance and visualise
    mixup = MyMixUp(alpha=1.0, seed=42)

    save_path = "./task2/mixup.png"
    # Create grid visualisation
    mixup.visualise_mixup_grid(train_dataset, num_samples=16, save_path=save_path)
    print(f"Saved visualisation to {save_path}")
