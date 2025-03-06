java c GR5242 HW04 Problem 5: Transfer learning with MobileNets For coding questions, look for filling in ########## Your code here ##########;  for reflection questions, write down your answers in the "Your Answer:" text block. There are many examples of external links to documentation. If you see reference to a pytorch or similar object, try hovering over the word to see if documentation is linked. Description: In this homework, you will practice (1 ) going over the full procedure of training a neural   network and (2) extending your knowledge on TF2, by implementing a transfer learning  task. You will incorporate the existing MobileNets to your own network structure and to   classify some new categories of images. Building and fitting the network from scratch is expensive and beyond the scope of this assignment, so you will load the MobileNets model which was pre-trained on the imagenet dataset. The version of MobileNet we're using is V2, which is a family of neural network architectures for efficient on-device image classification and related tasks. As a general advice, you can refer to the official documentations for more info if necessary. Import modules for later usage. In [ ]: import torch import torch.nn as nn import torchvision.transforms as transforms from torchvision.datasets import ImageFolder import matplotlib.pyplot as plt import numpy as np print("PyTorch version: ",torch.    version   ) In [ ]: if torch.cuda.is_available(): print("GPU is available.") else: print("GPU is not available.") Question 1: Download and format the data The data we are going to use is the Oxford flower dataset which consists of 102 categories of flowers. Each class consists of between 40 and 258 images. The images can be found here. The main difficulty of learninig from this dataset is in the large size of the classes. You may refer to this paper for what other researchers have done with it. (1) Data Pre-processing First, load the dataset from Kaggle (https://www.kaggle.com/competitions/oxford-102- flower-pytorch/data) where you can click " Download All" for the flower data. You can    also download it directly from the zip file provided. Then you split the data into training and testing sets. How many training and testing samples do you have? During the pre-processing stage, we would like to format all the images for the MobileNet module.For this module, the size of the input image is fixed to height x width = 224 x 224 pixels. The input images are expected to have 3 RGB color values in the range [0, 1 ], following   the common image input conventions (analogously to TF 1 .x).

In [ ]: #-----------------------------------------------------------

this part is not necessary
#for nn built-in flowers

#(note that the size of the dataset does match the tf dataset) #raw_train print(len(raw_train)) print(len(raw_test))# Access a specific data point (e.g., the 10th data point) index = 10  # Change this to the index you want to access  sample_image, label = raw_train [index]

Display the label and other information
print("nn raw data") print(f"Data at index {index}:") print(f"Label: {label}") print(f"Image shape: {sample_image.size}")

Apply the raw data transforms to raw_train and raw_test
train_nn = Flowers102(root= '', split="train", download=True, transform=trans test_nn = Flowers102(root= '', split="test", download=True, transform=transfo #for nn build-in flowers: raw_train print(len(train_nn)) print(len(test_nn)) sample_image, label = train_nn [100]

Display the label and other information
print("nn standardized image data") print(f"Data at index {index}:") print(f"Label: {label}") print(f"Image shape: {sample_image.shape}") (2) Data Exploration Let's plot some of the data.

from torch.utils.data import DataLoader, Dataset
assert isinstance(train_nn, Dataset) assert isinstance(test_nn, Dataset)

Print the datasets
print(train_nn) print(test_nn)

Reflection Question (1a): In the data exploration stage, what is the purpose of " assert isinstance(train, Dataset)"? Your Answer: Part 2: Self-defined CNN In this section, you will define your own CNN (convolutional neural network) to classify the Oxford flowers. Recall from the first problem, to build a neural network using  torch , we build a class that carries out the functions of the model, define an optimizer, and iterate through a few key steps. Here, we can make use of torch.nn.Sequential to save us a little hassle, now that we have seen how to build from the ground up in problem 1 . Instructions One suggestion is that you build a model with the following architecture, although you are free to try others as well with the same idea: 1 .) Convolution with 3x3 kernel, input shape is the image shape. Make use of torch.nn.Conv2d, followed by torch.nn.R代 写GR5242 HW04 Problem 5: Transfer learning with MobileNetsR 代做程序编程语言eLU and torch.nn.MaxPool2d with  kernel_size  2 and  stride  2 2.) Repeat step 1  (or a couple times), being careful about input shape 3.) Convolution with 3x3 kernel, input shape is the image shape. Make use of torch.nn.Conv2d, followed by torch.nn.ReLU and torch.nn.Flatten 4.) Fully connected layer using torch.nn.Linear and torch.nn.ReLU 5.) torch.nn.Dropout 6.) Linear layer returning us to number of classes (102) 7.) [ nothing ] or torch.nn.LogSoftmax to get label likelihood. Remember now that depending on which of these you use, you will need either criterion = nn.CrossEntropyLoss() or criterion = nn.NLLLoss() in training. If you use nn.CrossEntropyLoss() , you will need the extra step of calling nn.functional.softmax(output, dim=1) to compare outputs to targets in model evaluation, but not before calculating the loss in your training loop. After fitting the model, please test the accuracy of the prediction on the test set. In this stage, we do not ask for a great performance (you should have 'some' predictive performance though). But please ensure that you obtain a trainable model with no programming bugs. You may find it helpful to print the training progress bar or epoch.

Step 1: Model definition
Use a nn.Sequential model for deining your own CNN
########## Your code here ##########

Define the model using nn.Sequential, naming it model
Optional: print a summary of your model
from torchsummary import summary
Assuming your model_transfer is defined, you can print the summary
summary(our_model, (3, 224, 224))  # Assuming input size is (3, 224, 224)
Instructions: Here we will prepare ourselves for training. We need to define a few things before running our training loop, namely the  DataLoader ,  criterion , optimizer, and lr_scheduler . Instructions:

Fill in necessary blanks in the training loop, with the provided guidance Reflection Questions 2a:

(1 ) How did you choose your network structure? \ (2) Which optimizer did you use? Why? 
Your Answer: Part 3: Transfer Learning Using Pre-trained Model There are several types of transfer learning, as illustrated here. In this homework, you will practice B2, using MobileNet_V2. (1) Freeze the pre-trained model and fine-tune the transfer learning. Now you can go through the same steps to build and train the transfer learning model.

Instructions: Within the  model_transfer = nn.Sequential()  call, dd an Adaptive Average    Pooling layer with nn.AdaptiveAvgPool2d(), then perform. flattening and apply a linear layer as you should be familiar with from earlier. As before, remember your choice of whether to use Cross Entropy or Negative Log Likelihood, and make sure to use the corresponding output of your model (i.e., whether to apply Softmax after calculating loss or within the model) In [ ]: # Step 1: Model definition

Use a torch.nn Sequential model for defining the transfer learning model
Set MobileNetV2 parameters to non-trainable
for  param in  MobileNetV2.parameters(): param. requires_grad 〓  False

Use a custom reshape layer
class  ReshapeLayer(nn.Module): def  forward(self, x): return  x.view(x.size(0), x.size(1), 1, 1)

Create a Sequential model in PyTorch
model_transfer 〓  nn.Sequential( MobileNetV2, ReshapeLayer(),  # Reshape to [batch_size, num_channels, 1, 1] ########## Your code here ########## ) In [ ]: # define batch size here batch_size 〓  32 input_tensor 〓  torch. randn([batch_size, 3, 224, 224])

visualize the model graphical structure
#Iterate through the model and print the dimensions at each layer for  layer in  model_transfer: input_tensor 〓  layer(input_tensor)print(f"Layer: {layer.      class     .     name   }, Output Shape: {input_tensor.s In [ ]: #print(model_transfer) Instructions: As before, write code to define your  optimizer ,  DataLoader , (criterion) , and  lr_scheduler . Then, write a training loop.

Your code here should look similar to earlier in the assignment, outside of choosing hyperparameters, names, and possibly choice of loss. (2) Fine-tune some parameters in your network to see if you can improve the performance on testing data. (Optional) In [ ]: ########## Your code here ########## Reflection Questions 3a: (1 ) Briefly explain the network structure of MobileNet and how is it different from other models? (2) In your experiment, which parameter(s) is the network most sensitive to? Can you briefly reason why? (3) What are some pros and cons of doing transfer learning? (4) What is a batch? How does the batch size affect the training process? (5) What is an epoch during the training process? Your Answer: (6) Describe any observation you find interesting from the above experiment (Open- ended). Your Answer: In [ ]:

   加QQ codinghelp Email: cscholary@gmail.com
