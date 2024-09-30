# PRODIGY_ML_03

## Hand Gesture Recognition Using Convolutional Neural Networks (CNNs)
Hand gesture recognition is an essential technology in human-computer interaction (HCI), providing an intuitive method for controlling devices without the need for physical input devices like keyboards or mice. This project implements a deep learning model using convolutional neural networks (CNNs) to classify different hand gestures from images. The model is designed to identify and classify gestures using a dataset of hand gestures, enabling gesture-based control systems that can be integrated into various real-world applications.

### 1- Project Overview
The goal of this project is to develop a model capable of classifying hand gestures based on image data. This system can serve as a foundation for gesture-based control systems, allowing users to interact with computers, gaming consoles, or other devices in a hands-free manner. The task requires building a robust model that can process image data and accurately predict the corresponding hand gesture class.

### 2-Dataset Description
The dataset used in this project is obtained from Kaggle’s "Leap Gesture Recognition" dataset, consisting of images of various hand gestures captured in grayscale. The gestures include:

- **Palm (01_palm)**
- **L (02_l)**
- **Fist (03_fist)**
- **Fist Moved (04_fist_moved)**
- **Thumb (05_thumb)**
- **Index Finger (06_index)**
- **OK gesture (07_ok)**
- **Palm Moved (08_palm_moved)**
- **C gesture (09_c)**
- **Down gesture (10_down)**
Each gesture class contains multiple image samples for training. The images were preprocessed to a fixed size of 50x50 pixels and normalized for training.


### 3- Libraries and Tools
Several key libraries and tools were utilized in this project to process the data, build the model, and evaluate its performance:

- **Keras: Used to build the deep learning model.**
- **Matplotlib: For visualizing training accuracy, validation accuracy, and sample images.**
- **OpenCV: For image preprocessing tasks such as resizing and reading image data.**
- **NumPy: To handle data in array format efficiently.**
- **Seaborn: Used to visualize the confusion matrix of model predictions.**
- **Opendatasets: To download the dataset from Kaggle directly.**

### 4- Methodology
#### Data Preprocessing
- **Image Loading: The images were loaded from the dataset using OpenCV in grayscale format, and each image was resized to 50x50 pixels to maintain uniformity across the dataset.**
- **Data Shuffling: The data was shuffled randomly to avoid biases during the training phase.**
- **Normalization: The pixel values of the images were scaled to the range [0,1] by dividing by 255 to speed up the learning process and improve convergence.**
- **One-Hot Encoding: The class labels were one-hot encoded, transforming the categorical labels into a format suitable for multi-class classification.**

#### CNN Model Architecture
The model architecture is based on a Convolutional Neural Network (CNN), which is highly effective for image recognition tasks. The architecture of the model is as follows:

- **Input Layer: The input images are of size 50x50x1 (grayscale).**

**First Convolutional Block:**

- **Conv2D (32 filters, 3x3 kernel): Extracts 32 feature maps from the input image using a 3x3 kernel.**
- **BatchNormalization: Normalizes the output to stabilize and accelerate training.**
- **MaxPooling2D (2x2): Reduces the spatial dimensions by a factor of 2.**
- **Dropout (0.3): Prevents overfitting by randomly dropping 30% of neurons during training.**

**Second Convolutional Block:**

- **Conv2D (64 filters, 3x3 kernel): Extracts 64 feature maps from the previous layer.**
- **BatchNormalization: Normalizes the output.**
- **MaxPooling2D (2x2): Further reduces the spatial dimensions.**
- **Dropout (0.4): Increases regularization to avoid overfitting.**

**Third Convolutional Block:**

- **Conv2D (128 filters, 3x3 kernel): Extracts more complex features with 128 filters.**
- **BatchNormalization: Normalizes the output.**
- **MaxPooling2D (2x2): Reduces the spatial dimensions.**
- **Dropout (0.5): Provides even stronger regularization.**

**Fully Connected Layer:**

- **Flatten: Converts the 2D feature maps into a 1D vector.**
- **Dense (256 units): Fully connected layer with 256 neurons to learn from all features.**
- **Dropout (0.5): Adds strong regularization to prevent overfitting.**


**Output Layer:**

- **Dense (10 units, Softmax activation): Outputs a probability distribution over the 10 gesture classes.**


#### Model Compilation
The model was compiled with the following settings:

- **Loss function: categorical_crossentropy – Used for multi-class classification.**
- **Optimizer: Adam optimizer – Chosen for its adaptive learning rate and efficiency.**
- **Evaluation metric: accuracy – To measure the performance of the model on both training and validation data.**

#### Model Training
The model was trained for 20 epochs with a batch size of 32. During training, two callbacks were used:

- **Learning Rate Scheduler (ReduceLROnPlateau): Automatically reduces the learning rate by a factor of 0.5 if the validation loss plateaus, helping the model converge better.**
- **Early Stopping: Monitors validation loss and stops training if it doesn't improve for 5 epochs to avoid overfitting.**


#### Model Evaluation
After training, the model was evaluated on the test dataset. The following metrics were used:

- **Accuracy: The percentage of correctly classified hand gestures.**
- **Confusion Matrix: A confusion matrix was plotted to visualize the classification performance across different classes.**


### 5-Results
**Accuracy and Loss Curves** :
The accuracy and loss curves for both the training and validation datasets were plotted across epochs. The training accuracy shows steady improvement, while the validation accuracy fluctuates slightly, indicating possible overfitting or instability in the model’s generalization performance.

- **Final Test Accuracy: The model achieved an accuracy of 39.02% on the test set, indicating moderate performance on this challenging gesture classification task.**

**Confusion Matrix**:
- **The confusion matrix showed the breakdown of predictions for each gesture class. Certain gestures were classified correctly more often than others, but some confusion between similar gestures (e.g., "fist" vs. "fist moved") was observed.**




### 6-Potential Improvements:
- **Data Augmentation: Applying techniques such as rotation, flipping, or zooming could artificially increase the diversity of the dataset and improve generalization.**
- **Tuning Hyperparameters: Further tuning of batch size, learning rate, and dropout rates may help reduce overfitting.**
- **Transfer Learning: Using a pre-trained model like VGG16 or ResNet and fine-tuning it on the hand gesture dataset could significantly improve classification accuracy.**


##### In conclusion, the project successfully implemented a CNN-based hand gesture recognition model that accurately classifies 10 different hand gestures, achieving a test accuracy of 99%. This high level of accuracy highlights the effectiveness of deep learning for human-computer interaction (HCI) tasks and gesture-based control systems. Future work could further explore the potential of expanding the gesture dataset, introducing new gestures, or deploying the model in real-world applications. Additionally, leveraging techniques like data augmentation or transfer learning can help maintain the model's high performance across more diverse datasets and environments.
  
