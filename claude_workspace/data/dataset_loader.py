"""
Dataset loader for AI voice detection training.

Supports loading from directory structure or manifest files.
"""

import numpy as np
import csv
from pathlib import Path
from typing import Tuple, Optional, Generator, List, Dict
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from features import load_audio, pad_or_trim, normalize_audio, LFCCExtractor
from .augmentation import AudioAugmenter


class ManifestDatasetLoader:
    """
    Load dataset from a CSV manifest file.

    Manifest format:
        file_path,label,label_int,source
        /path/to/audio.flac,bonafide,0,asvspoof_la
        /path/to/audio2.flac,spoof,1,asvspoof_df
    """

    def __init__(
        self,
        manifest_path: str,
        sample_rate: int = config.SAMPLE_RATE,
        audio_duration: float = config.AUDIO_DURATION,
        feature_extractor: str = "lfcc",
        augment: bool = True,
    ):
        """
        Initialize ManifestDatasetLoader.

        Args:
            manifest_path: Path to CSV manifest file
            sample_rate: Target sample rate
            audio_duration: Audio window duration in seconds
            feature_extractor: Feature type ('lfcc' or 'mel')
            augment: Apply data augmentation
        """
        self.manifest_path = Path(manifest_path)
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.audio_samples = int(sample_rate * audio_duration)
        self.augment = augment

        # Initialize feature extractor
        if feature_extractor == "lfcc":
            self.extractor = LFCCExtractor()
        else:
            from features import MelSpectrogramExtractor
            self.extractor = MelSpectrogramExtractor()

        # Initialize augmenter
        self.augmenter = AudioAugmenter() if augment else None

        # Load manifest
        self.file_paths = []
        self.labels = []
        self._load_manifest()

    def _load_manifest(self):
        """Load file paths and labels from manifest."""
        with open(self.manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.file_paths.append(Path(row['file_path']))
                self.labels.append(int(row['label_int']))

        human_count = sum(1 for l in self.labels if l == 0)
        ai_count = sum(1 for l in self.labels if l == 1)
        print(f"Loaded manifest: {len(self.file_paths)} samples")
        print(f"  Human: {human_count}, AI: {ai_count}")

    def load_audio_file(self, file_path: Path) -> np.ndarray:
        """Load and preprocess audio file."""
        audio, _ = load_audio(file_path, self.sample_rate)
        audio = pad_or_trim(audio, self.audio_samples)
        audio = normalize_audio(audio)
        return audio

    def extract_features(self, audio: np.ndarray, augment: bool = False) -> np.ndarray:
        """Extract features from audio."""
        if augment and self.augmenter is not None:
            audio = self.augmenter(audio)
        return self.extractor.extract(audio)

    def load_all_data(
        self,
        max_samples: Optional[int] = None,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all data into memory.

        Args:
            max_samples: Limit total samples
            show_progress: Show progress bar

        Returns:
            Tuple of (features, labels)
        """
        file_paths = self.file_paths[:max_samples] if max_samples else self.file_paths
        labels = self.labels[:max_samples] if max_samples else self.labels

        features = []
        valid_labels = []

        iterator = tqdm(
            zip(file_paths, labels),
            total=len(file_paths),
            desc="Loading data"
        ) if show_progress else zip(file_paths, labels)

        for file_path, label in iterator:
            try:
                audio = self.load_audio_file(file_path)
                feat = self.extract_features(audio, augment=False)
                features.append(feat)
                valid_labels.append(label)
            except Exception as e:
                if show_progress:
                    print(f"Error loading {file_path}: {e}")

        return np.array(features), np.array(valid_labels)

    def split_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        val_split: float = config.VALIDATION_SPLIT,
        test_split: float = config.TEST_SPLIT,
        random_state: int = config.RANDOM_SEED,
    ) -> Tuple:
        """Split data into train/val/test sets."""
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            features, labels,
            test_size=test_split,
            random_state=random_state,
            stratify=labels,
        )

        val_ratio = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_trainval,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """Calculate class weights for imbalanced data."""
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        weights = {}
        for cls, count in zip(unique, counts):
            weights[cls] = total / (len(unique) * count)
        return weights


def create_data_generators_from_manifest(
    manifest_path: str,
    batch_size: int = config.BATCH_SIZE,
    val_split: float = config.VALIDATION_SPLIT,
    test_split: float = config.TEST_SPLIT,
    augment_train: bool = True,
    max_samples: Optional[int] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict]:
    """
    Create train, validation, and test data generators from manifest.

    Args:
        manifest_path: Path to CSV manifest file
        batch_size: Batch size
        val_split: Validation split ratio
        test_split: Test split ratio
        augment_train: Augment training data
        max_samples: Maximum total samples

    Returns:
        Tuple of (train_ds, val_ds, test_ds, info_dict)
    """
    loader = ManifestDatasetLoader(manifest_path, augment=augment_train)

    # Load all data
    features, labels = loader.load_all_data(max_samples=max_samples)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
        features, labels, val_split, test_split
    )

    # Create datasets
    train_ds = create_tf_dataset(X_train, y_train, batch_size, shuffle=True)
    val_ds = create_tf_dataset(X_val, y_val, batch_size, shuffle=False)
    test_ds = create_tf_dataset(X_test, y_test, batch_size, shuffle=False)

    # Info dict
    info = {
        "num_train": len(X_train),
        "num_val": len(X_val),
        "num_test": len(X_test),
        "class_weights": loader.get_class_weights(y_train),
        "input_shape": X_train.shape[1:],
    }

    print(f"\nDataset split:")
    print(f"  Train: {info['num_train']} samples")
    print(f"  Validation: {info['num_val']} samples")
    print(f"  Test: {info['num_test']} samples")
    print(f"  Input shape: {info['input_shape']}")

    return train_ds, val_ds, test_ds, info


class DatasetLoader:
    """
    Load and prepare datasets for training AI voice detection model.

    Expects data organized as:
    data_dir/
        human/
            file1.wav
            file2.wav
            ...
        ai/
            file1.wav
            file2.wav
            ...
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = config.SAMPLE_RATE,
        audio_duration: float = config.AUDIO_DURATION,
        feature_extractor: str = "lfcc",
        augment: bool = True,
        cache_features: bool = False,
    ):
        """
        Initialize DatasetLoader.

        Args:
            data_dir: Path to data directory
            sample_rate: Target sample rate
            audio_duration: Audio window duration in seconds
            feature_extractor: Feature type ('lfcc' or 'mel')
            augment: Apply data augmentation
            cache_features: Cache extracted features in memory
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.audio_samples = int(sample_rate * audio_duration)
        self.augment = augment
        self.cache_features = cache_features

        # Initialize feature extractor
        if feature_extractor == "lfcc":
            self.extractor = LFCCExtractor()
        else:
            from features import MelSpectrogramExtractor
            self.extractor = MelSpectrogramExtractor()

        # Initialize augmenter
        self.augmenter = AudioAugmenter() if augment else None

        # File cache
        self._file_cache: Dict[str, np.ndarray] = {}
        self._feature_cache: Dict[str, np.ndarray] = {}

        # Scan for files
        self.human_files = []
        self.ai_files = []
        self._scan_files()

    def _scan_files(self):
        """Scan data directory for audio files."""
        human_dir = self.data_dir / "human"
        ai_dir = self.data_dir / "ai"

        audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

        if human_dir.exists():
            self.human_files = [
                f for f in human_dir.iterdir()
                if f.suffix.lower() in audio_extensions
            ]

        if ai_dir.exists():
            self.ai_files = [
                f for f in ai_dir.iterdir()
                if f.suffix.lower() in audio_extensions
            ]

        print(f"Found {len(self.human_files)} human samples")
        print(f"Found {len(self.ai_files)} AI samples")

    def load_audio_file(self, file_path: Path) -> np.ndarray:
        """
        Load and preprocess audio file.

        Args:
            file_path: Path to audio file

        Returns:
            Preprocessed audio array
        """
        cache_key = str(file_path)

        if cache_key in self._file_cache:
            return self._file_cache[cache_key].copy()

        audio, _ = load_audio(file_path, self.sample_rate)
        audio = pad_or_trim(audio, self.audio_samples)
        audio = normalize_audio(audio)

        if self.cache_features:
            self._file_cache[cache_key] = audio

        return audio

    def extract_features(
        self,
        audio: np.ndarray,
        augment: bool = False,
    ) -> np.ndarray:
        """
        Extract features from audio.

        Args:
            audio: Audio array
            augment: Apply augmentation

        Returns:
            Feature array
        """
        if augment and self.augmenter is not None:
            audio = self.augmenter(audio)

        return self.extractor.extract(audio)

    def load_all_data(
        self,
        max_samples_per_class: Optional[int] = None,
        show_progress: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all data into memory.

        Args:
            max_samples_per_class: Limit samples per class
            show_progress: Show progress bar

        Returns:
            Tuple of (features, labels)
        """
        human_files = self.human_files[:max_samples_per_class]
        ai_files = self.ai_files[:max_samples_per_class]

        all_files = human_files + ai_files
        labels = [config.LABEL_HUMAN] * len(human_files) + \
                 [config.LABEL_AI] * len(ai_files)

        features = []
        valid_labels = []

        iterator = tqdm(all_files, desc="Loading data") if show_progress else all_files

        for file_path, label in zip(iterator, labels):
            try:
                audio = self.load_audio_file(file_path)
                feat = self.extract_features(audio, augment=False)
                features.append(feat)
                valid_labels.append(label)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        return np.array(features), np.array(valid_labels)

    def split_data(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        val_split: float = config.VALIDATION_SPLIT,
        test_split: float = config.TEST_SPLIT,
        random_state: int = config.RANDOM_SEED,
    ) -> Tuple:
        """
        Split data into train/val/test sets.

        Args:
            features: Feature array
            labels: Label array
            val_split: Validation split ratio
            test_split: Test split ratio
            random_state: Random seed

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train+val and test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            features, labels,
            test_size=test_split,
            random_state=random_state,
            stratify=labels,
        )

        # Second split: train and val
        val_ratio = val_split / (1 - test_split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_trainval,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced data.

        Args:
            labels: Label array

        Returns:
            Dict mapping class index to weight
        """
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)

        weights = {}
        for cls, count in zip(unique, counts):
            weights[cls] = total / (len(unique) * count)

        return weights


def create_tf_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = config.BATCH_SIZE,
    shuffle: bool = True,
    augmenter: Optional[AudioAugmenter] = None,
) -> tf.data.Dataset:
    """
    Create TensorFlow dataset from numpy arrays.

    Args:
        features: Feature array
        labels: Label array
        batch_size: Batch size
        shuffle: Shuffle data
        augmenter: Optional augmenter for training data

    Returns:
        tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_data_generators(
    data_dir: str,
    batch_size: int = config.BATCH_SIZE,
    val_split: float = config.VALIDATION_SPLIT,
    test_split: float = config.TEST_SPLIT,
    augment_train: bool = True,
    max_samples: Optional[int] = None,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict]:
    """
    Create train, validation, and test data generators.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size
        val_split: Validation split ratio
        test_split: Test split ratio
        augment_train: Augment training data
        max_samples: Maximum samples per class

    Returns:
        Tuple of (train_ds, val_ds, test_ds, info_dict)
    """
    loader = DatasetLoader(data_dir, augment=augment_train)

    # Load all data
    features, labels = loader.load_all_data(max_samples_per_class=max_samples)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(
        features, labels, val_split, test_split
    )

    # Create datasets
    train_ds = create_tf_dataset(X_train, y_train, batch_size, shuffle=True)
    val_ds = create_tf_dataset(X_val, y_val, batch_size, shuffle=False)
    test_ds = create_tf_dataset(X_test, y_test, batch_size, shuffle=False)

    # Info dict
    info = {
        "num_train": len(X_train),
        "num_val": len(X_val),
        "num_test": len(X_test),
        "class_weights": loader.get_class_weights(y_train),
        "input_shape": X_train.shape[1:],
    }

    print(f"\nDataset split:")
    print(f"  Train: {info['num_train']} samples")
    print(f"  Validation: {info['num_val']} samples")
    print(f"  Test: {info['num_test']} samples")
    print(f"  Input shape: {info['input_shape']}")

    return train_ds, val_ds, test_ds, info


class AudioDataGenerator:
    """
    Memory-efficient data generator for large datasets.

    Loads and processes audio files on-the-fly instead of loading
    all data into memory.
    """

    def __init__(
        self,
        file_paths: List[Path],
        labels: List[int],
        batch_size: int = config.BATCH_SIZE,
        sample_rate: int = config.SAMPLE_RATE,
        audio_duration: float = config.AUDIO_DURATION,
        augmenter: Optional[AudioAugmenter] = None,
        shuffle: bool = True,
    ):
        """
        Initialize AudioDataGenerator.

        Args:
            file_paths: List of audio file paths
            labels: List of labels
            batch_size: Batch size
            sample_rate: Target sample rate
            audio_duration: Audio duration in seconds
            augmenter: Optional augmenter
            shuffle: Shuffle data each epoch
        """
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.audio_samples = int(sample_rate * audio_duration)
        self.augmenter = augmenter
        self.shuffle = shuffle

        self.extractor = LFCCExtractor()
        self.indices = np.arange(len(file_paths))

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return int(np.ceil(len(self.file_paths) / self.batch_size))

    def on_epoch_end(self):
        """Shuffle indices at end of epoch."""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get batch by index."""
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.file_paths))
        batch_indices = self.indices[start_idx:end_idx]

        features = []
        batch_labels = []

        for idx in batch_indices:
            try:
                audio, _ = load_audio(self.file_paths[idx], self.sample_rate)
                audio = pad_or_trim(audio, self.audio_samples)
                audio = normalize_audio(audio)

                if self.augmenter is not None:
                    audio = self.augmenter(audio)

                feat = self.extractor.extract(audio)
                features.append(feat)
                batch_labels.append(self.labels[idx])
            except Exception:
                continue

        return np.array(features), np.array(batch_labels)

    def __iter__(self) -> Generator:
        """Iterate over batches."""
        for i in range(len(self)):
            yield self[i]
