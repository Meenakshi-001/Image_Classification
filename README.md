# Image_Classification
## About Project
In this project, We classify images of the following sports person:
1.Virat Kohli
2.Leonal Messi
3.Serena Williams
4.Roger Federer
5.Maria Sharapova

## Libraries used
1.Numpy
2.Mtplotlib
3.Tensor flow
4.cv2
5.tqdm
6.sci-kit learn

### Model Architecture:
The chosen model is a Convolutional Neural Network (CNN) implemented using TensorFlow and Keras. The architecture of the model is as follows:

    Convolutional layer with 32 filters and a kernel size of (3, 3), using ReLU activation.
    MaxPooling layer with a pool size of (2, 2).
    Flatten layer to convert the 2D feature maps into a 1D vector.
    Dense layer with 256 neurons and ReLU activation.
    Dropout layer with a dropout rate of 0.5 to prevent overfitting.
    Dense layer with 512 neurons and ReLU activation.
    Output layer with 5 neurons (equal to the number of classes) and softmax activation.

### Training Process:
The model is trained using the training set (70% of the data) with the following configurations:

    Optimizer: Adam
    Loss Function: Sparse Categorical Crossentropy (suitable for integer labels)
    Metrics: Accuracy
    Batch Size: 32
    Epochs: 50
    Validation Split: 30%

The training process involves feeding the training data through the network for a specified number of epochs, adjusting the weights based on the calculated loss, and validating the model's performance on a separate validation set.

### Critical Findings:

    The dataset consists of images of five different celebrities: Virat Kohli, Serena Williams, Roger Federer, Maria Sharapova, and Lionel Messi.
    The model achieves a certain level of accuracy on the validation set after 50 epochs.
    The accuracy on the test set is evaluated, and a classification report is generated, providing insights into the precision, recall, and F1-score for each class.
    The model is then used to predict the label of a single image from an external source.
