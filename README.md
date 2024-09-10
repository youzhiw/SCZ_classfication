# SCZ_classfication 

In this project, we focused on classifying schizophrenia using 2D MRI slices rather than full 3D MRI images. The motivation was to leverage deep learning techniques to detect schizophrenia-related patterns in brain scans efficiently. Our approach combined traditional CNN architectures with modern transformer models, aiming to create an explainable and generalizable classification system. We started by using a VGG model with squeeze-and-excitation (SE) blocks to extract significant slices from the 3D MRI scans. Grad-CAMs (Gradient-weighted Class Activation Mapping) were used to highlight the most crucial regions, helping narrow down the relevant slices for analysis.

Once the significant MRI slices were identified, they were fed into a SWIN Transformer model for classification. SWIN Transformers, which excel at capturing both local and global image features, were chosen for their ability to process medical images efficiently across varying resolutions. However, despite extensive experimentation with different configurations, we encountered difficulties in achieving reliable classification performance. Although our VGG model yielded an AUC of 0.81, a notable achievement for this initial phase, the SWIN Transformer’s performance did not surpass random guessing. Challenges such as the domain gap between ImageNet-pretrained models and medical imaging, as well as constraints in GPU resources, may have contributed to this result.

One of the key insights from this project was the importance of maintaining context in 3D MRI data. By relying on 2D slices, we may have missed critical spatial relationships inherent in 3D scans, limiting the model’s ability to fully comprehend the complexity of schizophrenia. Additionally, the Grad-CAM method, while useful for interpretability, may not have selected the most diagnostically relevant slices, further complicating the training of the SWIN Transformer. Another notable issue was the limitation of pretraining the SWIN model on non-medical images, which may have failed to capture the subtle features specific to brain MRIs.

You can find our final report here: https://github.com/youzhiw/SCZ_classfication/blob/main/BMI_Final_Report.pdf


Order of scripts to run:
- the data of the MRI images shoud be in a folder called 'data'
1. vgg_train.py
2. extract_gradCAM.py
3. train_transformer.py
