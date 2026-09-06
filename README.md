AI-Image-Detection

An AI-based image detection system developed as a Software Engineering graduation project.

The system analyzes images using multiple machine learning and deep learning approaches to determine whether an image is AI-generated (FAKE) or REAL.

Project Overview

The system combines three complementary detection approaches:

* EfficientNet-B0 — Deep learning-based image classification.
* JPEG/DCT Analysis — Frequency-domain analysis using Discrete Cosine Transform (DCT) features.
* Noise Consistency Analysis — XGBoost-based analysis of image noise and statistical patterns.

The final system combines the individual model predictions using a weighted ensemble and majority voting mechanism.

My Contribution

My main contribution to the project was the JPEG/DCT-based detection component.

The JPEG detector:

1. Converts the input image to grayscale.
2. Resizes the image to 128 × 128.
3. Applies the Discrete Cosine Transform (DCT).
4. Extracts the first 16 × 16 DCT coefficients.
5. Produces 256 frequency-domain features.
6. Applies feature scaling.
7. Uses a trained neural network classifier.
8. Calculates the probability that the image is AI-generated.

JPEG/DCT Pipeline

Input Image
     ↓
Grayscale Conversion
     ↓
Resize to 128 × 128
     ↓
Discrete Cosine Transform (DCT)
     ↓
Extract 16 × 16 Coefficients
     ↓
256 Features
     ↓
Feature Scaling
     ↓
Neural Network
     ↓
Fake / Real Probability
     ↓
Final Classification

Technologies

* Python
* PyTorch
* OpenCV
* NumPy
* Scikit-learn
* Joblib
* EfficientNet-B0
* XGBoost
* Discrete Cosine Transform (DCT)

Project Structure

AI-Image-Detection/
│
├── jpeg_dct_detector.py
├── requirements.txt
└── README.md

Installation

Clone the repository:

git clone https://github.com/karlabitar2/AI-Image-Detection.git
cd AI-Image-Detection

Install the required dependencies:

pip install -r requirements.txt

JPEG Detector Requirements

The JPEG/DCT detector requires the trained model and scaler files:

model_complete_gpu.pth
scaler_gpu.pkl

These trained model files are not included in the repository at the moment.

Place them in the project directory before running the detector.

Usage

Run the detector:

python jpeg_dct_detector.py

Enter the path of an image when prompted.

The detector returns:

* Fake probability
* Real probability
* Confidence score
* Final decision

Example:

Prediction:
{
    "model": "JPEG-DCT",
    "fake_probability": 0.82,
    "real_probability": 0.18,
    "confidence": 0.82,
    "decision": "FAKE"
}

System Architecture

The complete graduation project uses an ensemble approach:

                    Input Image
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
     JPEG / DCT     EfficientNet     XGBoost
       Detector        B0             Noise
          │              │              │
          ↓              ↓              ↓
      Prediction      Prediction     Prediction
          │              │              │
          └──────────────┼──────────────┘
                         ↓
                Weighted Ensemble
                         +
                  Majority Voting
                         ↓
                  Final Decision
                 FAKE / REAL

Important Note

This repository focuses on the JPEG/DCT component that I personally worked on as part of the larger graduation project.

The complete ensemble system was developed collaboratively as a graduation project.

Future Improvements

Possible future improvements include:

* Expanding the training dataset.
* Improving model generalization to unseen AI generators.
* Testing additional frequency-domain features.
* Adding explainability and visualization techniques.
* Deploying the detector as a web application.
* Comparing performance across different image-generation models.

Author

Karla Bitar

Software Engineer | AI & Automation | Data Analytics

GitHub: karlabitar2
