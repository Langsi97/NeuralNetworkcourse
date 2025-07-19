# Neural Network Project: Image Classification and Segmentation

This project explores neural network architectures for **image classification** and **image segmentation** tasks. Implemented using Python and deep learning frameworks, the notebook demonstrates steps from data preprocessing to model training and evaluation.

## 🧠 Project Overview

This notebook presents the application of artificial neural networks for solving:

- **Image Classification**: Assigning a label $y \in \{1, \dots, K\}$ to an input image $x \in \mathbb{R}^{H \times W \times C}$.
- **Image Segmentation**: Predicting a label for each pixel in the image, resulting in a mask $M \in \mathbb{R}^{H \times W}$.

The models are trained using categorical cross-entropy loss:

$$
\mathcal{L}(y, \hat{y}) = - \sum_{i=1}^K y_i \log(\hat{y}_i)
$$

## 🧪 Sample Code

```python
# Example: CNN model setup
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
