# SCZ_classfication 

In this project, we explored the use of 2D MRI slices to classify schizophrenia by combining traditional convolutional neural networks (CNN) and advanced transformer-based models. We aimed to develop an efficient and explainable model to detect schizophrenia using MRI data, focusing on using VGG models for feature extraction and SWIN Transformers for classification. Our strategy began by utilizing a modified VGG architecture with squeeze-and-excitation blocks to highlight significant regions in 3D brain MRIs using Grad-CAM. The resulting 2D slices were then used to train a SWIN Transformer.

The project offered valuable insights into the complexity of using deep learning for medical image classification, specifically in diagnosing mental health conditions like schizophrenia

Order of scripts to run:
- the data of the MRI images shoud be in a folder called 'data'
1. vgg_train.py
2. extract_gradCAM.py
3. train_transformer.py
