import numpy as np
import torch
from torchvision.utils import make_grid
from PIL import Image

class MyMixUp:
    def __init__(self, alpha=1.0, seed=42):
        self.alpha = alpha
        
        self.random_state = np.random.RandomState(seed)
        torch.manual_seed(seed)
    
    def __call__(self, x, y, device = 'cpu'):
        # x: input tensor Batch of images
        # y: target tensor Batch of labels
        if device == 'cuda':
            x = x.cuda()
            y = y.cuda()

        batch_size = x.size(0)
        
        if self.alpha > 0:
            lam = self.random_state.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # Create shuffled indices
        index = torch.randperm(batch_size, device=x.device)
        
        # Mix images
        mixed_x = lam * x + (1 - lam) * x[index, :]
        
        # Return mixed images and both sets of labels with lambda
        return mixed_x, y, y[index], lam
    
    def visualize_mixup_grid(self, dataset, num_samples=16, save_path="mixup.png", seed=42):
        """
        Visualize a montage of mixup augmented images and save to a PNG file.
        
        Args:
            dataset (torch.utils.data.Dataset): Dataset to sample images from
            num_samples (int): Number of mixed images to display. Default is 16.
            save_path (str): Path to save the visualization. Default is "mixup.png".
            seed (int, optional): Seed for reproducibility. Default is None.
            
        Returns:
            tuple: Mixed images, lambdas, original labels 1, original labels 2
        """
            
        # Get random samples from the dataset
        indices = np.random.choice(len(dataset), size=num_samples*2, replace=False)
        
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
                lam = np.random.beta(self.alpha, self.alpha)
            else:
                lam = 1
                
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
        
        # Convert from tensor to numpy and adjust range to 0-255 for PIL
        grid_np = grid.permute(1, 2, 0).cpu().numpy()
        
        # Convert numpy array to uint8 range for PIL
        grid_np = (grid_np * 255).astype('uint8')
        
        # Create PIL image and save
        img = Image.fromarray(grid_np)
        img.save(save_path)
        
        print(f"Grid visualization saved to {save_path}")
        
        return mixed_images, lambdas, labels1, labels2

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    def load_cifar10(batch_size=128):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    transform = transforms.Compose([
    transforms.ToTensor(),
    # Add other transformations if needed
])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    mixup = MyMixUp(alpha=0.2, seed=42)
    mixup.visualize_mixup_grid(train_dataset, num_samples=16, save_path="mixup.png")
