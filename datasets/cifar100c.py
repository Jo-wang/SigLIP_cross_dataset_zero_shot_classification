import os
import numpy as np
from .utils import DatasetBase
from .oxford_pets import OxfordPets
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]


    

    
class CIFAR100C(Dataset):
    dataset_dir = 'CIFAR-100-C'
    corruption = 'brightness' + '.npy'
    severity = 1
    def __init__(self, root):
        """
        Args:
            data_dir (str): Path to the CIFAR-100-C dataset.
            corruption_type (str): Type of corruption (e.g., "gaussian_noise").
            severity (int): Severity level of corruption (1 to 5).
            transform (callable, optional): Optional transform to be applied to the images.
        """
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, self.corruption)
        self.template = templates
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor()  # Convert PIL Image to PyTorch Tensor
        ])

        # Load corrupted images
        images_all_severities = np.load(self.image_dir)
        #  = np.load(os.path.join(data_dir, f"{corruption_type}.npy"))
        
        # Extract only the images for the specified severity
        num_images_per_severity = len(images_all_severities) // 5
        start_idx = (self.severity - 1) * num_images_per_severity
        end_idx = self.severity * num_images_per_severity
        self.images = images_all_severities[start_idx:end_idx]
        # Load labels
        self.labels = np.load(os.path.join(self.dataset_dir, "labels.npy"))[start_idx:end_idx]
        
        self.classnames = cifar100_classes
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert numpy array to PIL Image
        image = Image.fromarray(image)

        # Apply CLIP preprocessing
        if self.transform:
            image = self.transform(image)

        return image, label

# Load CIFAR-100 class names
cifar100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 
    'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 
    'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]