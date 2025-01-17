**Real vs AI Image Classification**
This repository implements a deep learning model using transfer learning (VGG16) to classify images as either real or AI-generated. The model leverages the power of convolutional neural networks (CNNs) and fine-tuning to achieve high accuracy on the given dataset.

**Features**
•	Transfer Learning: Uses the VGG16 model pre-trained on ImageNet as the base model.
•	Custom Layers: Adds custom dense and dropout layers for better classification performance.
•	Data Augmentation: Employs techniques like rotation, zoom, and flipping for robustness.
•	Training Pipeline: Includes a data pipeline for loading, preprocessing, and splitting the dataset.
•	Prediction Module: Predicts labels for test images and exports results to a CSV file.
•	Error Handling: Skips problematic images during feature extraction or prediction.

**Requirements**
Install the necessary libraries using pip:
pip install keras numpy pandas scikit-learn tqdm

**Training**
1.	Dataset Loading and Preprocessing:
o	The createdataframe function constructs a DataFrame with image paths and labels.
o	The extract_features function resizes images to 128x128 and normalizes pixel values.
2.	Model Architecture:
o	The model uses VGG16 as a base with its top layers removed.
o	Additional layers include GlobalAveragePooling2D, Dense layers, and Dropout for regularization.
3.	Fine-tuning:
o	Freezes most layers of VGG16 and unfreezes the last 8 layers for fine-tuning.
4.	Training:
o	Runs the training loop with data augmentation and callbacks:
	ReduceLROnPlateau: Reduces the learning rate on plateau.
	ModelCheckpoint: Saves the best-performing model.
Run the training script:
python train_classifier.py

**Testing**
1.	Prediction Functionality:
o	test_on_1_img(img_name): Predicts the label for a single image.
o	test(dir): Processes all images in a directory and generates predictions.
2.	CSV Output:
o	Generates two CSV files:
	prediction.csv: Raw predictions.
	prediction_sorted.csv: Predictions sorted by numeric IDs.
Run the test script:
python test_classifier.py

**Key Functions**
createdataframe(dir)
Creates a DataFrame with image paths and labels from the dataset directory.
extract_features(images)
Preprocesses images by resizing, normalizing, and handling problematic files.
test_on_1_img(img_name)
Loads and preprocesses a single image, predicts its label, and maps it back to the original class.
test(dir)
Processes all images in a directory and saves predictions to CSV files.

________________________________________





**GAN for MNIST Digit Generation**
This repository implements a Generative Adversarial Network (GAN) in PyTorch to generate MNIST digit images. The GAN consists of a Generator that creates images from random noise and a Discriminator that distinguishes real images from fake ones.

**Features**
•	Generator: Maps latent vectors to 28×28 MNIST-like images using fully connected layers and activation functions.
•	Discriminator: Determines whether a given 28×28 image is real or fake.
•	Dataset: Uses the MNIST dataset, normalized to [-1, 1], for training.
•	Loss: Binary Cross Entropy Loss with logits for both Generator and Discriminator.
•	Visualization: Displays real and generated images during training.

**Requirements**
To run this code, you need the following dependencies:
•	Python 3.7+
•	PyTorch
•	torchvision
•	matplotlib
•	tqdm
Install the required Python packages using pip:
pip install torch torchvision matplotlib tqdm

**Requirements**
**Training Loop**
1.	Train the Discriminator to:
o	Maximize the loss for fake images (outputs near 0).
o	Minimize the loss for real images (outputs near 1).
2.	Train the Generator to:
o	Fool the Discriminator by maximizing its output for fake images (outputs near 1).

**Results**
During training, real and generated images are displayed every few iterations to visualize the Generator's progress.






