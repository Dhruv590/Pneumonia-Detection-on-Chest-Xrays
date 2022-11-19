# Introduction
## Motivation –
Pneumonia is the inflammation of tissues in one or both lungs, usually caused by a bacterial 
infection. It affects many individuals, especially children, primarily in developing and 
underdeveloped countries characterized by risk factors such as overcrowding, poor hygienic 
conditions, malnutrition, and the unavailability of appropriate medical facilities. Early 
diagnosis of pneumonia is crucial to cure the disease completely. More than 1 million people 
in the USA are hospitalized annually with pneumonia, and fifty thousand die from this illness
[1]. Chest X-ray images are the best-known and standard clinical method for diagnosing 
pneumonia. 
## Problem Statement –
Streptococcus pneumoniae, a type of bacteria, is a common cause of Pneumonia, a potentially 
fatal infectious disease that can attack one or both of a person's lungs. According to the World 
Health Organization (WHO), Pneumonia is to blame for one in three deaths in India. 
Radiotherapists with advanced training are required to evaluate chest X-rays, in order to 
diagnose pneumonia. Therefore, creating an automatic system for diagnosing pneumonia will
help treat the condition quickly, especially in remote areas.
## Challenges –
Examination of X-ray scans is the most common means of diagnosis. However, diagnosing 
pneumonia from chest X-ray images is challenging for even expert radiologists. The 
appearance of pneumonia in X-ray images is often unclear, can confuse with other diseases, 
and can behave like many other benign abnormalities. These inconsistencies caused many 
subjective decisions and varieties among radiologists in diagnosing pneumonia. Thus, an 
automatic computer-aided design system (in this case a deep learning model) with generalizing 
capability is required to diagnose the disease [2].

# Related Work and Contributions –

Okeke S. et al. [3] performed a study on the large dataset containing the train, validation & test, 
and two sub-folders include Pneumonia and non-Pneumonia class. The proposed model was 
made from scratch to the end that extracts into two parts: a feature extractor & a classifier. In 
the pre-processing part, data augmentation was applied. The model consisted of several Conv. 
layers with max-pooling and Relu activation. During the training period, different types of 
output sizes were tested and combined into a 1D feature for the classifier procedure. While the 
classifier, the dropout (0.5), 2 dense layers with Relu and output with the Sigmoid activation 
were implemented. The final results were that, among several image shapes, the best shape was 
(200 x 200 x 3) and it has a validation score - 94% with a loss of 0.18%. The study encountered 
that the larger the image size, the less validation score, vice versa, the smaller size images have 
performed well [4].
Mohammad Farukh Hashmi et al [5] experimented on a Large Dataset of Labeled Optical 
Coherence Tomography (OCT) and Chest X-Ray Images [6] which has 5858 images. Dataset 
was labelled by two physicians and evaluated by another physician. 5156 images used the 
training set and 700 images in the test set. Viral and bacterial pneumonia is considered 
pneumonia infected. Normal/no pneumonia are augmented twice to remove imbalance data in 
the training set. Different neural networks expect images of different sizes according to their 
defined architecture, so images were resized to 224*224 & 229*229. All 5 models i.e. 
ResNet18, DenseNet121, and MobileNetV2, InceptionV3 and Xception were trained 
individually and then a weighted sum was taken to classify. SGD as the optimizer was used as 
it has better generalization, and the model was trained for 25 epochs. The learning rate, the 
momentum, and the weight decay were set to 0.001, 0.9, and 0.0001, respectively. Case1 -
Every model contributed equally towards the final predictions i.e. 0.20 Weight of each 
architecture was considered. Test accuracy of 97.45 and a loss of 0.087 was obtained. In case2-
optimum weights were taken based on individual architecture classification performance, 
found out the final weighted classifier was able to achieve a testing accuracy of 98.43, and the 
testing loss was 0.062 with an AUC of 99.76 [4].

# Data –

The dataset that we will be using for this project is taken from Kaggle, and can be found here 
Link. The dataset is organized into 3 folders (train, test, validation) and contains subfolders for 
each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 
categories (Pneumonia/Normal). Chest X-ray images (anterior-posterior) were selected from 
retrospective cohorts of paediatric patients of one to five years old from Guangzhou Women 
and Children’s Medical Centre, Guangzhou. All chest X-ray imaging was performed as part of 
patients’ routine clinical care. For the analysis of chest x-ray images, all chest radiographs were 
initially screened for quality control by removing all low quality or unreadable scans. The 
diagnoses for the images were then graded by two expert physicians before being cleared for 
training the AI system. In order to account for any grading errors, the evaluation set was also 
checked by a third expert [7].

# Proposed Solution and Current Results –

## Solution Approach –
As Deep Learning algorithms have gained popularity in medical image analysis, Convolution 
Neural Networks (CNNs) have become a valuable tool for disease classification. In this project, 
we aim to develop a robust CNN model/pipeline from scratch that can be utilized to extract 
features from chest X-rays to classify abnormal and normal X-Rays. Furthermore, pre-trained 
CNN models can help us classify images using features learned from large-scale datasets. 
Statistical results demonstrate that supervised classifier algorithms coupled with pre-trained 
CNN models are highly beneficial in detecting pneumonia in chest X-ray images. Hence, we 
also aim to utilize pre-trained CNN models (Transfer Learning) followed by different 
classifiers for classifying normal and abnormal chest X-Rays. In order to determine the optimal 
pre-trained CNN model for the purpose, we will conduct an analytical analysis between various 
Transfer Learning models (such as VGG19, InceptionV3 (GoogLeNet), ResNet50, 
EfficientNet etc). In the end, we plan to conduct a comparative study between the model that 
we trained from scratch and different transfer learning models clubbed with supervised 
classifier algorithms that we will use in this project. 
## Current Work and Results –
The data set we are working with is unevenly split between normal and abnormal (pneumoniaaffected) chest X-Rays. Out of 5,863 chest X-ray images, 1,583 are typical lung X-Ray images, 
and 4,273 are lung X-Ray images with Pneumonia. 

![image](https://user-images.githubusercontent.com/48237615/202839245-a116a7cf-927c-4ff8-9712-6b62958eb034.png)

In order to tackle the problem of data imbalance, we used a common technique known as Data Augmentation. The basic idea behind Data Augmentation is to alter the training data with minor transformations to modify the array representation while preserving the label. Grayscales, vertical and horizontal flips, random cropping, translations, and rotations are some of the widely used augmentation techniques. We can easily double or quadruple the number of training examples and build a robust model by applying only a few adjustments to our training data. The raw images present in our dataset had different sizes from one another. Hence, we reduced the size of every mage to 150 x 150 in order to introduce uniformity and save computation power. 

![image](https://user-images.githubusercontent.com/48237615/202839256-aa8c2306-fabe-41ae-99cf-dd6bd3706ec5.png)

After data pre-processing, we fed the images to a Convolutional Neural Network having the following structure – 

![image](https://user-images.githubusercontent.com/48237615/202839269-0942e00d-6d8f-4fcd-a23f-f25dbafee109.png)

The model was trained for 20 epochs, with ReduceLROnPlateau (to vary the learning rate) having a patience value of 2, factor of 0.3, and a min_lr of 0.000001. The following graphs shows the train and validation accuracy and loss over 20 epochs. 

![image](https://user-images.githubusercontent.com/48237615/202839279-bd9b1842-b692-4d18-b3c1-1e3d9456e8f1.png)

The best validation accuracy of the model was achieved at the 4th epoch. Hence, moving forward, we will be using the weights attained in the 4th epoch. Upon predicting the test data using the weights attained in the 4th epoch, the following results were obtained.

![image](https://user-images.githubusercontent.com/48237615/202839285-31ca3b5a-84ba-4d48-827f-f0f83f696ecb.png)

Since, there was a data imbalance in the dataset, we will be using F1 score (takes into account precision and recall) as the evaluation metric. We obtain an f1 score of 90% with the model we trained from scratch. The confusion matrix for the prediction is as follows – 

![image](https://user-images.githubusercontent.com/48237615/202839294-b4faaa2a-bf81-4213-8a53-834d755e592e.png)

Following images shows some of the chest X-rays that were incorrectly classified – 

![image](https://user-images.githubusercontent.com/48237615/202839298-da7d9790-8eb7-418c-a877-82d976c56cf7.png)

We will use this model as a baseline model, and we plan to further enhance this model before moving on to using Transfer Learning models. Further, we will use various Transfer Learning models and compare their performance with other transfer learning model as well as the model we will obtain after enhancing the baseline model.

# References –

[1] W. H. Organization, "Standardization of interpretation of chest radiographs for the diagnosis of pneumonia in children", 2001.

[2] M. I. Neuman et al., "Variability in the interpretation of chest radiographs for the diagnosis of pneumonia in children", Journal of hospital medicine, vol. 7, no. 4, pp. 294-298, 2012.

[3] Stephen, O., Sain, M., Maduh, U. J., & Jeong, D.-U. (2019). An efficient deep learning approach to pneumonia classification in healthcare. Journal of Healthcare Engineering, 2019, 4180949.

[4] Narayana Darapaneni, Ashish Ranjan, Dany Bright “Pneumonia Detection in Chest X-Rays using Neural Networks”.

[5] M. F. Hashmi, S. Katiyar, A. G. Keskar, N. D. Bokde, and Z. W. Geem, “Efficient pneumonia detection in chest X-ray images using deep transfer learning,” Diagnostics (Basel), vol. 10, no. 6, p. 417, 2020

[6] D. Kermany, K. Zhang and M. Goldbaum, "Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images", Mendeley, 2021. [Online] https://data.mendeley.com/datasets/rscbjbr9sj/3. [Accessed: 01- May- 2021].

[7] https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia




