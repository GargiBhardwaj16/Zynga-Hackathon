# Team : VisionQuest
# Intelligent Image Matching System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Libraries](https://img.shields.io/badge/Libraries-Numpy%2C%20OpenCV%2C%20TensorFlow%2C%20PyTorch-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen)

Welcome to the **Intelligent Image Matching System**! This project is designed to develop an AI-powered algorithm capable of accurately matching provided screenshots from a game scene to a set of similar-looking images, with an emphasis on both accuracy and efficiency.

## 🚀 Problem Statement

### Objective

Develop an algorithm that can:
1. 🖼️ Identify the main image from two test images within a set.
2. 🔍 Extract key visual features and contextual elements from game scenes.
3. 📊 Return the similarity percentage of the main image relative to its matching counterpart in the test images.
4. 💰 Identify the total win amount in a specific game scene for Set 8.
5. 🎰 Identify the bet amount in a specific game scene for Set 9.

### Game Reference

🎮 [Willy Wonka Slots](https://play.google.com/store/apps/details?id=com.zynga.wonka&hl=en_IN&pli=1)

## Data Sets

- **Sets 1 to 7**: Each set contains 3 images - `Image.png` (Main Image), `Test1.png` (Test Image 1), `Test2.png` (Test Image 2).
- **Sets 8 and 9**: Each set contains 2 images - `Test1.png` (Test Image 1), `Test2.png` (Test Image 2).

## Input Format

- **Sets 1 to 7**: 
  - Return similarity percentages: `[percentage_for_test1, percentage_for_test2]`.
- **Set 8**:
  - Return the total win amount: `[total_win_amount_test1, total_win_amount_test2]`.
- **Set 9**:
  - Return the bet amount: `[bet_amount_test1, bet_amount_test2]`.

## 📂 Folder Structure

The `Problems` folder contains all the sets:
- [Problems Folder](https://drive.google.com/drive/folders/1VQTbSl_NdygxhlEs0aUUkj0qDUtyAROL?usp=sharing)

## 🛠️ Getting Started

### Prerequisites

- 🐍 Python 3.x
- 📦 Necessary libraries: `numpy`, `pandas`, `opencv-python`, `scikit-learn`, `tensorflow` or `pytorch`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/intelligent-image-matching.git
2. Navigate to the project directory:
   ```bash
   cd intelligent-image-matching
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
## Usage
Place your images in the Problems folder.

Check the output for similarity percentages, total win amounts, and bet amounts as per the sets.
## 🧠 Algorithm Approach


1. Feature Extraction: Use techniques like SIFT, SURF, or deep learning-based feature extractors.
2. Matching Algorithm: Employ algorithms like K-Nearest Neighbors (KNN) or deep learning-based models to find the best match.
3. Similarity Calculation: Compute similarity scores using metrics like cosine similarity or Euclidean distance.
4. Contextual Analysis: Extend the model to recognize text and numerical values in specified regions for Sets 8 and 9.


## 🤝 Contributing
We welcome contributions to enhance the accuracy and efficiency of the algorithm. Feel free to fork the repository, make your changes, and submit a pull request.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
Inspired by the image matching challenges in computer vision.
Special thanks to the creators of Willy Wonka Slots for providing the game scenes.
Let's match those images with precision and efficiency! 🎮🔍

## Note 
For detailed information on the implementation, please refer to the Documentation section.
