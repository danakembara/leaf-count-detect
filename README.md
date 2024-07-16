# Leaf Counting and Detection
Deep Learning Course (GRS-4806)  
Dana Putra Kembara and Efraim Manurung  
Wageningen University & Research  
2023<br>
<img src="https://github.com/user-attachments/assets/dc47b632-9e38-41ce-941c-2f0d9f36f82a" alt="image" width="750"/>

## Introduction
Plant phenotype refers to the set of observable characteristics resulting from interactions between gene expression and the environment. Accurate and efficient monitoring of plant phenotypes is essential for intelligent production in plant cultivation, leading to better decision-making and higher yields. Traditional plant monitoring relies on manual measurements to analyze phenotypes, which are labor-intensive, time-consuming, and subject to observer bias. A better approach is to use image-based plant phenotyping with deep learning, allowing remote observation and reducing the effects of manual interference. Convolutional neural network (CNN)-based methods have been commonly applied to plant phenotyping, with related works such as [1] which used CNN for leaf counting, and [2][3] which used CNN for leaf detection.

In this project, we will use datasets from the Leaf Counting Challenge (LCC) and the Leaf Segmentation Challenge (LSC) to perform two tasks: leaf counting and leaf detection. We will compare and evaluate the performance of several architectures, including ObjectDetectorMultiScale, AlexNet, pre-trained ResNet18, pre-trained ResNet34, pre-trained ResNet50, and VGG16. Additionally, we will conduct experiments with data augmentation, varying hyperparameters, and Freezing Weights (FW) to enhance the robustness of the architectures and achieve the best performance for both tasks. Finally, we will compare our results with existing results from the literature.

## Methods
### Dataset
Our datasets are sourced from the ‘Ara2012’ and ‘Ara2013-Canon’ sets from the Leaf Segmentation Challenge (LSC) and Leaf Counting Challenge (LCC), which contain top-down images of Arabidopsis plants along with ground-truth annotations. One of the main challenges we face is the limited amount of data available for training. The original dataset contains roughly 150 images for the training set and 50 for the test set. To address this issue and avoid overfitting, we employ data augmentation techniques to artificially increase our dataset. The augmented dataset is created by:
- Randomly cropping an area of 70% to 90% of the original size, with the ratio of width to height of the region randomly selected between 1 and 2
- Randomly changing the brightness and saturation of the image to a value between 80% and 90%, with a contrast adjustment of up to 150% of the original images
- Applying a horizontal flip with a 50% probability

### Task 1: Leaf Counting
For the leaf counting task, we developed a CNN regression network and trained it on the training set. We use three different networks for comparison:
- AlexNet with hyperparameters: batch size = 256, image size = 128, learning rate = 10^-4, and epochs = 35
- Fine-tuned ResNet18 with hyperparameters: batch size = 256, image size = 128, learning rate = 5x10^-5, and epochs = 50
- Fine-tuned ResNet50 with hyperparameters: batch size = 256, image size = 128, learning rate = 5x10^-5, and epochs = 50

We define Mean Squared Error (MSE) as the loss function and use Adam as the optimizer. As this is a regression task, the performance of these models is evaluated by calculating their Pearson correlation coefficient (r) on the test set. The closer the value is to 1 or -1, the closer the network predictions are to the ground truth.

### Task 2: Leaf Detection
For the leaf detection task, we developed a CNN detection network and trained it on the training set. We use four different networks for comparison: 
- A simple object detector called ObjectDetectorMultiScale using hyperparameters batch size = 32, image size = 256, learning rate = 10-4, epochs = 35, and weight decay = 10-4. We used different experiments for this network:
  - Using original data.
  - Using augmented data.
  - Using a variation of image size = [128, 256, 384]
  - Using a variation of learning rate = [10-2, 10-3, 10-4]
  - Using a variation of epochs = [20, 35, 50]
  - Using a variation of weight decay = [10-2, 10-3, 10-4]
-	ResNet18 using hyperparameters batch size = 32, image size = 128, learning rate = 10-4, epochs = 35, and weight decay = 10-4. We implement the model with and without freezing weights (FW)
-	ResNet34 using hyperparameters batch size = 32, image size = 128, learning rate = 10-4, epochs = 35, and weight decay = 10-4. We implement the model with and without freezing weights (FW)
-	VGG16 using hyperparameters batch size = 32, image size = 128, learning rate = 10-4, epochs = 35, and weight decay = 10-4. We implement the model with and without freezing weights (FW)

We also define Cross Entropy as the loss function and Adam as the optimizer. The performance of these models is evaluated by calculating their Average Precision (AP) on the test set. The closer the value is to 1, the closer the network predictions are to the ground truth. As we have a lot of overlapping objects, we change the non-maximum suppression threshold for calculating the AP to 0.5 and making predictions to 0.3.

## Results and Discussions
### Task 1: Leaf Counting
The training of the three models (AlexNet, pre-trained ResNet18, and pre-trained ResNet50) on the dataset is implemented in Python 3.6 and PyTorch as the back-end using Google Colab. As mentioned in the methods section, we use the data augmentation technique to mitigate the small amount of our train data to avoid overfitting. Figure 1 shows the differences between original and augmented images. For experiment purposes, we trained the original and augmented datasets separately to all models, which took us between 0.5 and 1 hour to converge, as training graphs shown in Figure 2. The Pearson correlation coefficient (r) then evaluates their performance.
Results from Table 1 demonstrate that AlexNet using the augmented dataset has given the best performance over other models with r = 0.76, an improvement from AlexNet when using the original dataset with r = 0.74. The alleged boost in performance comes from data augmentation applications. An experiment conducted by Gomes and Zheng using the same LSC and LCC datasets for leaf counting tasks showed a 2% improvement in performance metrics after data augmentation [1]. They also stated that data augmentation is significant for regularizing the models trained on limited datasets by generalizing out-of-sample images, which explains why there is an increase in performance.
However, based on our observation, data augmentation is not a one-size-fits-all solution to increase performance. Results from Table 1 demonstrate that ResNet50 using the original dataset, performs better than the augmented one. According to Gomes and Zheng, data augmentations are effective but can be too domain-specific [1]. The original dataset could fit better than the augmented dataset based on the characteristic of the ResNet50 model combined with our small training dataset. But if we analyze more, the ResNet50 using the augmented dataset converges faster, as shown in Figure 2. If the performance metrics also include the inference time, using the augmented data is better than using the original one. Also, the current results may change if we add new training images, alter the data augmentation technique, or change the hyperparameters.
Furthermore, results from Table 1 demonstrate that AlexNet performance is superior to pre-training ResNet18 and ResNet50. Unlike AlexNet, we use ResNet as a pre-trained model using fine-tuning from the ImageNet dataset to transfer learning with our training dataset. The pre-trained model can be an alternative solution in combating a small training dataset, as it already has well-balanced weights from an extensive dataset. However, according to Poojary et al., the main limitation of pre-trained models is they are built with standard models, so making changes to this model may severely affect the performance of the models [4]. We modify the last layer of the ResNet model by flattening it to solve a regression task, and this could lead to lower performance which explains why AlexNet is better in this case.
In conclusion, we use several CNN models to do a leaf counting task and find that the best performance model is AlexNet using data augmentation with r = 0.76. When challenging small train datasets, data augmentation has proven can increase performance considerably. However, using different data augmentation techniques may change the results. AlexNet also performs better than pre-training ResNet, as pre-trained models perform less when we modify the model to solve this specific regression task. Future research is necessary to improve model performance by modifying the data augmentation, using other models, and increasing the size and variability training dataset with new images.

### Task 2: Leaf Detection
The training of the various models and their experiments (ObjectDetectorMultiScale, pre-trained ResNet18, pre-trained ResNet34, pre-trained VGG16) on the dataset is implemented in Python 3.6 and PyTorch as the back-end using Google Colab. The learning process for whole process took about between 1 and 1.5 hour to converge.
By using augmented data we can increase the AP with small datasets, The authors of [2] proposed a data augmentation method, preserving the photorealistic appearance of plant leaves and we implemented it on the methods. 
Upon analyzing the data presented in both Figure 3, it is evident that data augmentation has been applied to the original dataset. The deep learning model, ObjectDetectorMultiScale, has been trained using both the original and augmented data, and the results have been recorded in Figure 4 and Table 3. The Average Precision (AP) then evaluates their performance. It is noteworthy that the performance of the model has improved with the use of augmented data, highlighting the importance of data augmentation in deep learning applications. Specifically, when using the original dataset without augmentation, the model achieved a AP score of 0.33, whereas with augmented data, its performance increased to 0.37, representing an improvement of 0.04 or 4%. The augmentation techniques used in this study were limited to random cropping and adjusting brightness.
In our experiment on leaf detection using deep learning, we also tested various hyperparameters and the results in Table 2. We found that using the original dataset without augmentation and an image size of 128 pixels yielded the highest average precision of 0.32. Additionally, a learning rate of 0.0001 resulted in an AP of 0.33, while training for 35 epochs gave an AP of 0.34. Lastly, a weight decay of 0.0001 was found to be optimal, resulting in an AP of 0.30. Based on these results, we have chosen these hyperparameters for use in architectures that involve encoders and Freezing Weights (FW). 
The results presented in Table 3 show the average precision scores for different pre-trained deep learning architectures with augmented data. Among the various encoder architectures, ResNet34 demonstrated the best performance, achieving an average precision score of 0.38, followed by ResNet18 with a score of 0.31, and VGG16 with a score of 0.30. Additionally, FW for ResNet18 and ResNet34 resulted in AP scores of 0.26 and 0.23, respectively. However, VGG16 with FW outperformed the other architectures with a score of 0.38, which is the highest average precision score achieved in this experiment and most reliable. These findings suggest that VGG16 with FW could be a use for object detection tasks that have small datasets. 
In terms of performance on small leaf datasets, it is possible that ResNet18 and ResNet34 might outperform YOLO, as they are designed for image classification tasks and have a lower computational cost compared to YOLO. However, YOLO has the advantage of being specifically designed for object detection tasks and may have better performance in detecting small objects, such as leaves [3].
Our experiments faced several limitations that affected the accuracy of our models. One significant limitation was the small size of our dataset, which may have limited the ability of our models to learn complex features and patterns. As a result, we may not have achieved the highest possible accuracy with the VGG16FW model. Additionally, we had limited GPU resources, which prevented us from deploying more complex models with higher parameters such as ResNet50. 
In conclusion, we use several CNN variation models to do a leaf detection task and find that the best performance model is VGG16FW using data augmentation with AP = 0.38. Despite the limitations our project demonstrated the effectiveness of Non-maximum Suppression (NMS) in improving the accuracy of object detection models. By using a threshold of 0.5, NMS was able to differentiate overlapping objects and achieve significant improvement in our models. Tuning the NMS threshold can also become an important parameter to consider when training object detection models. In addition, using more data augmentation techniques can increase the dataset variations and improve the training process. Going forward, increasing the size of the dataset can potentially improve the accuracy of the model.

## References

## Figures and Tables
<img width="450" alt="image" src="https://github.com/user-attachments/assets/cd9b7616-d945-4faa-8715-ad58e64e2969"><br>

<img width="750" alt="image" src="https://github.com/user-attachments/assets/7665a469-b82e-4a04-864c-7dc3f9b8eb67">



