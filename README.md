# Leaf Counting and Detection

![image](https://github.com/user-attachments/assets/dc47b632-9e38-41ce-941c-2f0d9f36f82a)

## Introduction
Plant phenotype is a set of observable characteristics resulting from interactions between gene expression and the environment. Accurate and efficient monitoring of plant phenotype is essential for intelligent production in plant cultivation, which leads to better decision-making resulting in a higher yield. Traditional plant monitoring still uses manual measurement to analyze phenotypes, which are labor-intensive, time-consuming, and biased toward the observer. A better approach is to use image-based plant phenotyping using deep learning that allows distant observation and reduces the effects of manual interference. Convolutional neural network (CNN)-based methods have commonly been applied to plant phenotyping. Related works are from [1] that used CNN for leaf counting, and [2][3] that used CNN for leaf detection.
__
In this project, we will use datasets from Leaf Counting Challenge (LCC) and Leaf Segmentation Challenge (LSC) as input to do the two tasks, leaf counting and leaf detection. The architectures that we use are ObjectDetectorMultiScale, AlexNet, pre-trained ResNet18, pre-trained ResNet34, pre-trained ResNet50, and VGG16 to be compared and evaluate their performances. We also implemented some experiments with data augmentation, a variation of hyperparameters, and Freezing Weights (FW), to make the architectures more robust and get the best performance for both tasks. Lastly, we compare the performance with existing results from papers.

