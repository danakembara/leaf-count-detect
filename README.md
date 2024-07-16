# Leaf Counting and Detection
- Deep Learning Course (GRS-4806)
- Dana Putra Kembara & Efraim Manurung
- Wageningen University & Research
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

## Results, Discussions, and Conclusions
### Task 1: Leaf Counting
The training of three models (AlexNet, pre-trained ResNet18, and pre-trained ResNet50) on the dataset was conducted using Python 3.6 and PyTorch on Google Colab. As described in the methods section, we employed data augmentation to mitigate the small size of our training data and avoid overfitting. Figure 1 illustrates the differences between original and augmented images. For experimental purposes, we separately trained the models on the original and augmented datasets, each taking between 0.5 to 1 hour to converge, as depicted in the training graphs shown in Figure 2. The Pearson correlation coefficient (r) was used to evaluate their performance.
<p align="center">
  <img width="450" alt="image" src="https://github.com/user-attachments/assets/cd9b7616-d945-4faa-8715-ad58e64e2969">
  <br>
  <img width="700" alt="image" src="https://github.com/user-attachments/assets/7665a469-b82e-4a04-864c-7dc3f9b8eb67">
  <br>
  <img width="425" alt="image" src="https://github.com/user-attachments/assets/6e74c36f-82f9-426a-933c-1a34602645f2">
</p>

Results from Table 1 demonstrate that AlexNet achieved the best performance using the augmented dataset with r = 0.76, showing an improvement over AlexNet trained on the original dataset with r = 0.74. This enhancement can be attributed to the effective application of data augmentation techniques. Previous studies by [1] using the LSC and LCC datasets also reported a similar improvement of 2% after employing data augmentation. However, it is important to note that the effectiveness of data augmentation can be context-specific. For instance, Table 1 reveals that ResNet50 performed better with the original dataset than with the augmented one. According to [1], while data augmentation generally enhances model generalization, its impact can vary based on the model's characteristics and dataset specifics. Nevertheless, our observation also indicates that ResNet50 trained with the augmented dataset converges faster, as shown in Figure 2. When considering inference time along with performance metrics, using augmented data proves advantageous over the original dataset.  

Moreover, Table 1 shows that AlexNet outperforms both pre-trained ResNet18 and ResNet50. Unlike AlexNet, ResNet models were initialized with weights from the ImageNet dataset and fine-tuned on our task-specific dataset. This approach leverages transfer learning to mitigate the challenges posed by small training datasets. However, as noted by [4], modifying pre-trained models for specific tasks can sometimes lead to decreased performance due to alterations in model architecture. In our case, modifying the last layer of ResNet models for regression tasks might explain why AlexNet exhibits superior performance.

In conclusion, our study employed various CNN models for leaf counting and found that AlexNet achieved the highest performance with r = 0.76 when augmented data was used. Data augmentation proved effective in enhancing performance, especially with limited training data. However, the choice of data augmentation techniques can significantly impact results, and the adaptability of different models should be considered for optimal performance in specific tasks. Future research should focus on refining data augmentation strategies, exploring alternative models, and expanding training datasets with new images.

### Task 2: Leaf Detection
The training of various models (ObjectDetectorMultiScale, pre-trained ResNet18, ResNet34, VGG16) and their experiments were conducted using Python 3.6 and PyTorch on Google Colab. The training process took between 1 to 1.5 hours to converge. Data augmentation played a crucial role in improving Average Precision (AP) scores for models trained on small datasets. Authors in [2] proposed a data augmentation method that preserved the photorealistic appearance of plant leaves, which we implemented in our methods. 
<p align="center">
  <img width="455" alt="image" src="https://github.com/user-attachments/assets/3696af08-6e78-4192-8c36-93e8d991613d">
  <br>
  <img width="442" alt="image" src="https://github.com/user-attachments/assets/5fb6b9a7-05f6-4c41-ad72-e38efb92bc2f">
  <br>
  <img width="368" alt="image" src="https://github.com/user-attachments/assets/eed8ff19-05d6-4aad-b80c-4e4521e58ab4">
  <br>
  <img width="259" alt="image" src="https://github.com/user-attachments/assets/03c1864d-bb87-457b-b235-6eae5bf3dfeb">
</p>

Upon analyzing the images presented in Figure 3, it is evident that data augmentation was applied to the original dataset. The ObjectDetectorMultiScale model was trained on both original and augmented data. Our augmentation techniques included random cropping and brightness adjustments. In our leaf detection experiments using deep learning, we also explored various hyperparameters as summarized in Table 2, with results detailed in Figure 4 and Table 3. AP was used to evaluate model performance, and significant improvements were observed when using augmented data. Specifically, the model achieved an AP score of 0.37 with augmented data, compared to 0.33 with the original dataset, marking a 4% increase in performance.

The highest AP of 0.32 was achieved using the original dataset without augmentation and an image size of 128 pixels. Additionally, a learning rate of 0.0001 resulted in an AP of 0.33, and training for 35 epochs yielded an AP of 0.34. Optimal performance with a weight decay of 0.0001 resulted in an AP of 0.30. Based on these findings, these hyperparameters were selected for architectures involving encoders and Freezing Weights (FW). Table 3 presents AP scores for different pre-trained deep learning architectures using augmented data. Among these, ResNet34 demonstrated the best performance with an AP score of 0.38, followed by ResNet18 at 0.31 and VGG16 at 0.30. Implementing FW for ResNet18 and ResNet34 resulted in AP scores of 0.26 and 0.23, respectively. Notably, VGG16 with FW outperformed other architectures with an AP score of 0.38, indicating its effectiveness in object detection tasks with small datasets.

Comparatively, ResNet18 and ResNet34 may perform better than YOLO in handling small leaf datasets, given their lower computational cost and suitability for image classification tasks [3]. However, YOLO's design specifically for object detection may offer superior performance in detecting small objects like leaves. Our experiments faced limitations, particularly due to the small dataset size, which may have constrained our models' ability to learn complex features. Additionally, limited GPU resources prevented us from deploying more complex models such as ResNet50, potentially affecting the accuracy of the VGG16FW model.

In conclusion, our study demonstrated the effectiveness of VGG16FW with data augmentation, achieving the highest AP score of 0.38 for leaf detection. Non-maximum Suppression (NMS) played a crucial role in improving model accuracy by distinguishing overlapping objects. Tuning the NMS threshold remains a critical parameter for training robust object detection models. Expanding data augmentation techniques and increasing dataset size are recommended for future research to further enhance model accuracy.

## References
[1] Gomes, D. P. S., and L. Zheng. 2020. Recent data augmentation strategies for deep learning in plant phenotyping and their significance. engrxiv.org/t3q5p [online]  
[2] Buzzy, M.; Thesma, V.; Davoodi, M.; Mohammadpour Velni, J. 2020. Real-Time Plant Leaf Counting Using Deep Object Detection Networks. Sensors, 20, 6896. https://doi.org/10.3390/s20236896  
[3] Dmitry Kuznichov, Alon Zvirin, Yaron Honen, Ron Kimmel. 2019. Data Augmentation for Leaf Segmentation and Counting Tasks in Rosette Plants. Computer Vision and Pattern Recognition.  https://doi.org/10.48550/arXiv.1903.08583  
[4] Poojary, R., R. Raina., and A. K. Mondal. 2021. Effect of data-augmentation on fine-tuned CNN model performance. International Journal of Artificial Intelligence, Volume 10, No. 1, pp. 84-92. https://doi.org/10.11591/ijai.v10.i1.pp84-92
