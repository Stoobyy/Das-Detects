"""
CNN-based Audio Classifier for AI vs Human Voice Detection
===========================================================
Uses Mel spectrograms as input features and a CNN architecture
to classify audio as AI-generated or human voice.
"""

import os
import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass
import warnings
import pickle

# Silence librosa warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    sample_rate: int = 16000
    duration: float = 3.0  # seconds
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    max_samples: Optional[int] = None  # Limit samples per class for quick testing


class AudioFeatureExtractor:
    """Extract Mel spectrogram features from audio files."""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.target_length = int(config.sample_rate * config.duration)
    
    def load_audio(self, file_path: str) -> Optional[np.ndarray]:
        """Load and preprocess audio file."""
        try:
            # Load audio at target sample rate
            audio, sr = librosa.load(file_path, sr=self.config.sample_rate, mono=True)
            
            # Pad or truncate to fixed length
            if len(audio) < self.target_length:
                audio = np.pad(audio, (0, self.target_length - len(audio)), mode='constant')
            else:
                audio = audio[:self.target_length]
            
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract Mel spectrogram from audio."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.sample_rate,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        return mel_spec_norm
    
    def process_file(self, file_path: str) -> Optional[np.ndarray]:
        """Process a single audio file and return Mel spectrogram."""
        audio = self.load_audio(file_path)
        if audio is None:
            return None
        return self.extract_mel_spectrogram(audio)


class DatasetLoader:
    """Load and prepare dataset for training."""
    
    def __init__(self, ai_dir: str, human_dir: str, config: AudioConfig):
        self.ai_dir = Path(ai_dir)
        self.human_dir = Path(human_dir)
        self.config = config
        self.extractor = AudioFeatureExtractor(config)
    
    def get_audio_files(self, directory: Path) -> List[Path]:
        """Get all audio files from a directory."""
        extensions = ['.wav', '.flac', '.mp3', '.ogg', '.m4a']
        files = []
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))
        return files
    
    def load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and process all audio files."""
        print("Loading dataset...")
        
        ai_files = self.get_audio_files(self.ai_dir)
        human_files = self.get_audio_files(self.human_dir)
        
        # Limit samples if specified
        if self.config.max_samples:
            ai_files = ai_files[:self.config.max_samples]
            human_files = human_files[:self.config.max_samples]
        
        print(f"Found {len(ai_files)} AI files and {len(human_files)} human files")
        
        features = []
        labels = []
        
        # Process AI files (label = 0)
        print("Processing AI voice files...")
        for i, file_path in enumerate(ai_files):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(ai_files)} AI files")
            mel_spec = self.extractor.process_file(str(file_path))
            if mel_spec is not None:
                features.append(mel_spec)
                labels.append(0)  # AI = 0
        
        # Process Human files (label = 1)
        print("Processing Human voice files...")
        for i, file_path in enumerate(human_files):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(human_files)} human files")
            mel_spec = self.extractor.process_file(str(file_path))
            if mel_spec is not None:
                features.append(mel_spec)
                labels.append(1)  # Human = 1
        
        X = np.array(features)
        y = np.array(labels)
        
        # Add channel dimension for CNN (samples, height, width, channels)
        X = X[..., np.newaxis]
        
        print(f"Dataset loaded: {X.shape[0]} samples, feature shape: {X.shape[1:]}")
        return X, y


def create_cnn_model(input_shape: Tuple[int, int, int]) -> keras.Model:
    """Create a CNN model for audio classification."""
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense Layers
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model


class VoiceClassifier:
    """Main classifier for AI vs Human voice detection."""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.model = None
        self.extractor = AudioFeatureExtractor(self.config)
        self.history = None
    
    def train(self, ai_dir: str, human_dir: str, 
              epochs: int = 50, 
              batch_size: int = 32,
              validation_split: float = 0.2,
              model_save_path: str = "voice_classifier_model.keras"):
        """Train the CNN model."""
        
        # Load dataset
        loader = DatasetLoader(ai_dir, human_dir, self.config)
        X, y = loader.load_dataset()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Create model
        input_shape = X_train.shape[1:]
        self.model = create_cnn_model(input_shape)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        print("\nStarting training...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            class_weight={0: 1.0, 1: len(y[y==0]) / len(y[y==1])}  # Handle class imbalance
        )
        
        # Evaluate on test set
        print("\n" + "="*50)
        print("Final Evaluation on Test Set")
        print("="*50)
        
        test_loss, test_acc, test_auc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        
        # Detailed classification report
        y_pred = (self.model.predict(X_test) > 0.5).astype(int).flatten()
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['AI', 'Human']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print("              Predicted")
        print("             AI   Human")
        print(f"Actual AI    {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"Actual Human {cm[1][0]:4d}  {cm[1][1]:4d}")
        
        # Save config
        config_path = model_save_path.replace('.keras', '_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(self.config, f)
        print(f"\nConfig saved to {config_path}")
        
        return self.history
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        self.model = keras.models.load_model(model_path)
        
        # Load config if exists
        config_path = model_path.replace('.keras', '_config.pkl')
        if os.path.exists(config_path):
            with open(config_path, 'rb') as f:
                self.config = pickle.load(f)
                self.extractor = AudioFeatureExtractor(self.config)
        
        print(f"Model loaded from {model_path}")
    
    def predict(self, audio_path: str) -> Tuple[str, float]:
        """
        Predict whether an audio file is AI-generated or human.
        
        Returns:
            Tuple of (label, confidence)
            - label: 'AI' or 'Human'
            - confidence: float between 0 and 1
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        # Extract features
        mel_spec = self.extractor.process_file(audio_path)
        if mel_spec is None:
            raise ValueError(f"Could not process audio file: {audio_path}")
        
        # Add batch and channel dimensions
        mel_spec = mel_spec[np.newaxis, ..., np.newaxis]
        
        # Predict
        prediction = self.model.predict(mel_spec, verbose=0)[0][0]
        
        if prediction > 0.5:
            return 'Human', float(prediction)
        else:
            return 'AI', float(1 - prediction)
    
    def predict_batch(self, audio_paths: List[str]) -> List[Tuple[str, float]]:
        """Predict multiple audio files."""
        results = []
        for path in audio_paths:
            try:
                label, confidence = self.predict(path)
                results.append((path, label, confidence))
            except Exception as e:
                results.append((path, 'Error', str(e)))
        return results


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train CNN for AI vs Human voice detection"
    )
    parser.add_argument(
        "--ai-dir", "-a",
        default="ai",
        help="Directory containing AI voice samples"
    )
    parser.add_argument(
        "--human-dir", "-u",
        default="human",
        help="Directory containing human voice samples"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--model-path", "-m",
        default="voice_classifier_model.keras",
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=3.0,
        help="Audio duration in seconds"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per class (for quick testing)"
    )
    parser.add_argument(
        "--predict", "-p",
        type=str,
        default=None,
        help="Path to audio file to predict (requires trained model)"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = AudioConfig(
        duration=args.duration,
        max_samples=args.max_samples
    )
    
    # Create classifier
    classifier = VoiceClassifier(config)
    
    if args.predict:
        # Prediction mode
        classifier.load_model(args.model_path)
        label, confidence = classifier.predict(args.predict)
        print(f"\nPrediction for: {args.predict}")
        print(f"  Label: {label}")
        print(f"  Confidence: {confidence:.2%}")
    else:
        # Training mode
        classifier.train(
            ai_dir=args.ai_dir,
            human_dir=args.human_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_save_path=args.model_path
        )


if __name__ == "__main__":
    main()
