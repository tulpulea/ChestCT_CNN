Project 2 Writeup

Data Source: https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

The following is the write-up for this personal project completed by Archan Tulpule.

Project Description: Chest Tumor Image Classification with Tensor Flow

Skills demonstrated: Python, classification, image data, tensorflow - keras, Convolutional neural networks, bias-variance tradeoff, hyper-paramter tuning, normalizing data, augmenting training data, checkpointing model weigths, drop-out regularization

Solution description: 
This image classification project was done to showcase my skills and understanding of basic neural networks in the form of CNN's. The data source had provided seperate training, validation, and testing data samples. The tensorflow pipeline greatly simplifies the process of designing, training, tuning, and comparing neural networks. First data was normalized, such that pixel values were brought within the range of 0 to 1. Then training data was augmented - a crucial step to reduce model variance. The model then included various covolutional layers and max pooling layers before ending with a softmax layer.The standard ReLu was the activation function of choice.

Plots were made at various points to visualize the progress of fitting with respect to the number of training epochs. One major observation made was that the testing data seemingly didn't adequately represent the ground truth. The evidence for this was seen as the final model fitting saw validation accuracy over 80%, but the testing accuracy was a disappointingly low 40%. 

However, this is a relatively arbitrary problem, since in real-world applications, data should be accumualated, shuffled, and split by the data scientist so as to prevent poor splits such as this.

After making this simple change by accumulating the data with the bash script copy_move.sh, and then essentially repeating the entire process, a test accuracy of 86% was achieved, and a very "textbook" training and testing error graph was created.

The full project is available on main.ipynb and a shortened version with the main code is available on main.py.

