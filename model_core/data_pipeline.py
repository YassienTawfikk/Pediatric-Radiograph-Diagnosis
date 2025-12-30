import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

# Reproducibility
SEED = 42
np.random.seed(SEED)

class DataPipeline:
    """
    Manages data loading, preprocessing, and augmentation for pneumonia detection.
    """
    
    def __init__(self, base_path, img_size=(224, 224), batch_size=32):
        self.base_path = Path(base_path)
        self.img_size = img_size
        self.batch_size = batch_size
        self.stats = {}
        
        # ImageNet normalization
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])
        
        print(f"{'='*70}\nData Pipeline Initialized\n{'='*70}")
        print(f"📁 Base Path: {self.base_path}")
        print(f"📐 Image Size: {self.img_size}")
        print(f"📦 Batch Size: {self.batch_size}\n{'='*70}")
    
    def explore_dataset(self):
        """Load and analyze dataset structure."""
        print(f"\n{'─'*70}\n📊 EXPLORING DATASET\n{'─'*70}")
        
        if not self.base_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.base_path}")
        
        # Count images per split and class
        splits = ['train', 'test', 'val']
        stats = {}
        
        for split in splits:
            split_path = self.base_path / split
            if split_path.exists():
                stats[split] = self._count_images(split_path)
        
        self.stats = stats
        self._print_statistics()
        return stats
    
    def _count_images(self, path):
        """Count images in each class folder."""
        class_counts = {}
        for class_folder in path.iterdir():
            if class_folder.is_dir():
                images = list(class_folder.glob('*.jpeg')) + \
                        list(class_folder.glob('*.jpg')) + \
                        list(class_folder.glob('*.png'))
                class_counts[class_folder.name] = len(images)
        return class_counts
    
    def _print_statistics(self):
        """Display dataset statistics."""
        print("\n📈 Dataset Statistics:")
        total = 0
        
        for split, classes in self.stats.items():
            if classes:
                split_total = sum(classes.values())
                total += split_total
                print(f"\n{split.upper()}:")
                for cls, count in classes.items():
                    pct = (count / split_total * 100)
                    print(f"  • {cls}: {count:,} ({pct:.1f}%)")
                print(f"  → Total: {split_total:,}")
        
        print(f"\n{'='*70}\n🎯 Total Images: {total:,}\n{'='*70}")
    
    def create_validation_split(self, val_ratio=0.15):
        """
        Create stratified train/validation split.
        
        Args:
            val_ratio: Proportion for validation (default: 15%)
        
        Returns:
            Train and validation image lists with labels
        """
        print(f"\n{'─'*70}\n📂 CREATING VALIDATION SPLIT ({int(val_ratio*100)}%)\n{'─'*70}")
        
        train_path = self.base_path / 'train'
        if not train_path.exists():
            raise FileNotFoundError(f"Training directory not found: {train_path}")
        
        # Collect all images and labels
        images, labels = [], []
        for class_folder in train_path.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name
                class_images = list(class_folder.glob('*.jpeg')) + \
                              list(class_folder.glob('*.jpg')) + \
                              list(class_folder.glob('*.png'))
                images.extend(class_images)
                labels.extend([class_name] * len(class_images))
        
        # Stratified split
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            images, labels,
            test_size=val_ratio,
            random_state=SEED,
            stratify=labels
        )
        
        # Verify no leakage
        overlap = set(img.name for img in train_imgs) & set(img.name for img in val_imgs)
        status = "✅ PASS" if len(overlap) == 0 else f"⚠️ FAIL ({len(overlap)} overlaps)"
        
        print(f"✓ Split: {len(train_imgs):,} train | {len(val_imgs):,} validation")
        print(f"✓ Leakage Check: {status}")
        
        # Display distribution
        self._print_split_distribution(train_labels, val_labels)
        
        self.train_imgs = train_imgs
        self.val_imgs = val_imgs
        return train_imgs, val_imgs, train_labels, val_labels
    
    def _print_split_distribution(self, train_labels, val_labels):
        """Print class distribution after split."""
        print("\n📊 Class Distribution:")
        
        train_dist = Counter(train_labels)
        val_dist = Counter(val_labels)
        
        print("Training:")
        for cls, count in train_dist.items():
            print(f"  • {cls}: {count:,} ({count/len(train_labels)*100:.1f}%)")
        
        print("Validation:")
        for cls, count in val_dist.items():
            print(f"  • {cls}: {count:,} ({count/len(val_labels)*100:.1f}%)")
    
    def create_generators(self, use_augmentation=True):
        """
        Create data generators with optional augmentation.
        
        Returns:
            train_generator, val_generator, test_generator
        """
        print(f"\n{'─'*70}\n🔄 CREATING DATA GENERATORS\n{'─'*70}")
        
        # Training augmentation
        if use_augmentation:
            print("✓ Training: WITH augmentation")
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                fill_mode='nearest'
            )
        else:
            print("✓ Training: WITHOUT augmentation")
            train_datagen = ImageDataGenerator(rescale=1./255)
        
        # Validation/test: no augmentation
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_gen = train_datagen.flow_from_directory(
            str(self.base_path / 'train'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            color_mode='rgb',
            shuffle=True,
            seed=SEED
        )
        
        val_path = self.base_path / 'val'
        val_gen = val_test_datagen.flow_from_directory(
            str(val_path),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            color_mode='rgb',
            shuffle=False,
            seed=SEED
        ) if val_path.exists() else None
        
        test_gen = val_test_datagen.flow_from_directory(
            str(self.base_path / 'test'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            color_mode='rgb',
            shuffle=False,
            seed=SEED
        )
        
        print(f"✓ Generators created:")
        print(f"  • Training batches: {len(train_gen)}")
        if val_gen:
            print(f"  • Validation batches: {len(val_gen)}")
        print(f"  • Test batches: {len(test_gen)}")
        print(f"  • Class mapping: {train_gen.class_indices}")
        
        return train_gen, val_gen, test_gen
    
    def visualize_samples(self, num_samples=5):
        """Display sample images from each class."""
        train_path = self.base_path / 'train'
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        
        for idx, class_folder in enumerate(train_path.iterdir()):
            if class_folder.is_dir() and idx < 2:
                images = list(class_folder.glob('*.jpeg'))[:num_samples]
                for i, img_path in enumerate(images):
                    img = load_img(img_path, target_size=self.img_size)
                    axes[idx, i].imshow(img, cmap='gray')
                    axes[idx, i].axis('off')
                    if i == 0:
                        axes[idx, i].set_title(
                            class_folder.name.upper(),
                            fontsize=12, fontweight='bold'
                        )
        
        plt.suptitle('Sample Chest X-Rays', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
