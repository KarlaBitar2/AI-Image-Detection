"""
JPEG/DCT-Based Fake Image Detector

This module extracts JPEG/DCT frequency-domain features from an image
and uses a trained neural network classifier to estimate whether
the image is AI-generated or real.
"""

import os
import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn


class ImageDetectorNN(nn.Module):
    """Neural network architecture used by the JPEG/DCT detector."""

    def __init__(self, input_size: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))


class JPEGDetector:
    """
    JPEG/DCT-based image detector.

    The detector:
    1. Converts the image to grayscale.
    2. Resizes it to 128x128.
    3. Applies the Discrete Cosine Transform (DCT).
    4. Extracts the first 16x16 DCT coefficients.
    5. Scales the resulting 256 features.
    6. Uses a trained neural network to classify the image.
    """

    def __init__(
        self,
        model_path="model_complete_gpu.pth",
        scaler_path="scaler_gpu.pkl"
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.name = "JPEG-DCT"

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler file not found: {scaler_path}"
            )

        # Load feature scaler
        self.scaler = joblib.load(scaler_path)

        # Load trained model checkpoint
        checkpoint = torch.load(
            model_path,
            map_location=self.device
        )

        input_size = checkpoint["input_size"]

        # Create model architecture
        self.model = ImageDetectorNN(input_size).to(self.device)

        # Load trained weights
        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

    def extract_features(self, image):
        """
        Extract DCT-based frequency features.

        Returns:
            numpy.ndarray: 256 DCT features.
        """

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )
        else:
            gray = image

        # Resize image
        resized = cv2.resize(
            gray,
            (128, 128)
        )

        # Normalize pixel values
        image_float = np.float32(resized) / 255.0

        # Apply Discrete Cosine Transform
        dct = cv2.dct(image_float)

        # Extract first 16x16 coefficients
        features = dct[:16, :16].flatten()

        return features

    def predict(self, image_path):
        """
        Predict whether an image is fake or real.

        Args:
            image_path (str): Path to the image.

        Returns:
            dict: Prediction probabilities and decision.
        """

        if not os.path.exists(image_path):
            return {
                "error": f"Image not found: {image_path}"
            }

        try:
            # Load image
            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                return {
                    "error": "Unable to read image."
                }

            # Extract DCT features
            features = self.extract_features(image)

            # Scale features
            features_scaled = self.scaler.transform(
                [features]
            )

            # Convert to PyTorch tensor
            tensor = torch.FloatTensor(
                features_scaled
            ).to(self.device)

            # Model prediction
            with torch.no_grad():
                probability = self.model(
                    tensor
                ).item()

            # Classification
            is_fake = probability > 0.5

            fake_probability = probability
            real_probability = 1 - probability

            confidence = (
                fake_probability
                if is_fake
                else real_probability
            )

            decision = (
                "FAKE"
                if is_fake
                else "REAL"
            )

            return {
                "model": self.name,
                "is_fake": bool(is_fake),
                "fake_probability": float(
                    fake_probability
                ),
                "real_probability": float(
                    real_probability
                ),
                "confidence": float(confidence),
                "decision": decision
            }

        except Exception as error:
            return {
                "error": f"Prediction error: {str(error)}"
            }


if __name__ == "__main__":

    print("=" * 60)
    print("JPEG/DCT Fake Image Detector")
    print("=" * 60)

    detector = JPEGDetector()

    image_path = input(
        "Enter image path: "
    ).strip()

    result = detector.predict(image_path)

    print("\nPrediction:")
    print(result)
