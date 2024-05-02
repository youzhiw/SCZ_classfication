# Schiz_Classification_BMEN4460_DLBI

### This repository is private, only sharing with students doing BMEN 4460 course final project in Spring 2024 at Columbia.
## Project name: Schizophrenia MRI Classification Using Deep Learning

All required source codes and commands are saved in this repositor, including:

**main_10fold.py**

**data_loader_10fold.py**

**GlobalLocalTransformer3D_multiscale.py**

**vgg3D_for_transformer.py**

**vgg.py**

**command.sh** (command instruction file, not code)

**Improving Across-Dataset Schizophrenia Classification with Structural Brain MRI Using Multi-scale Transformer.pdf** (the model raw abstract to help you understand the whole project, optional reading material) (Abstract is accepted by ISMRM 2023 and is going to be published in June)

If you are familiar with deep learning pipelines and have Python coding experience, you should have no trouble understanding the codes on your own. If you do need help implementing them, please see the instructions for each file below:


### Before we start: ###

All T1w MRI data are stored on Google drive in 10 folds here: https://drive.google.com/drive/folders/1hnYGHeXUGB4GM7IPrCGRon3bTnwf_NwJ?usp=share_link.

All the codes are saved in .py format. If you are using online coding enviroment such as Colab and Kaggle, you probably run into the jupyter notebooks ending with .ipynb. To run the .py files on Colab, please see this  [quick tutorial](https://rafat-joy99.medium.com/how-to-run-py-files-on-google-colab-46af2831e166/) and more helpful instructions online.

The .py codes used to be runned on the server with a cuda GPU. If you are utilizing public available GPUs such as Colab GPU, you may need to set up your cuda, otherwise the codes may give you an error about cuda. See this [discussion](https://stackoverflow.com/questions/50560395/how-to-install-cuda-in-google-colab-gpus) as an example on how to check and set up your cuda. Hopefully it should be fast and straightforward to set up your cuda index and name. These kinds of GPU errors can pop up differently for different laptops and Google account situations. Please first try to solve it on your own, as this problem may happen all the time when you are running any machine learning codes (including the midterm), even beyond our course.

### main_10fold.py ###

The only .py file you need to run using the command in command.sh. It imports the dataloader and the model, runs the main functions (and saves the trained model).

Please change the data path to where you save all ten folds of your MRI data! (there is a upper-cased notice at where you need to change the path)

Please modify the 'device' variable based on how you set up / name your cuda!

Please take a thorough look into all the 'parser.add_argument' to have a basic idea of what arguments and parameters we have (and we can change).

You can loaded a pre-trained model, or train a new model. The models can be but not limited to:
1. 'GlobalLocalBrainAge', the transformer-based model, as default. (It is named as BrainAge model, but has been major modified for our classification task)
2. VGG model, commented out for now.


### GlobalLocalTransformer3D_multiscale.py ###

The 3D Multi-scale Transformer (MST) model code.

To undersnad the Multi-Scale Transformer better, read the [abstract](https://github.com/TianYe10/Schiz_Classification_BMEN4460_DLBI/blob/main/Improving%20Across-Dataset%20Schizophrenia%20Classification%20with%20Structural%20Brain%20MRI%20Using%20Multi-scale%20Transformer.pdf)! Especially the figure 1!

All unnecessary codes are commented out to save your time. If you want to modify the structure, you may want to take a deeper look at line [204](https://github.com/TianYe10/Schiz_Classification_BMEN4460_DLBI/blob/83ed0b675615844ab313e5f7bbc98bb37b8ef26d/GlobalLocalTransformer3D_multiscale.py#L204) ~ 381.

### vgg3D_for_transformer.py ###

This is the vgg backbone for 3D MST model! Please leave this file as it be, don't touch it. It's similar to the actual vgg.py model, but again, it just serves for GlobalLocalTransformer3D_multiscale.py, and is not the file you want to import for vgg. ***To import vgg, you will import from vgg.py.***


### data_loader_10fold.py ###

Another important code file you want to go through other than the main_10fold.py. Two most important functions in it is the [__init__](https://github.com/TianYe10/Schiz_Classification_BMEN4460_DLBI/blob/efbe48ced140878aa04d2dfeaf1ac74b8923b0f2/data_loader_10fold.py#L18), and [_getitem_](https://github.com/TianYe10/Schiz_Classification_BMEN4460_DLBI/blob/efbe48ced140878aa04d2dfeaf1ac74b8923b0f2/data_loader_10fold.py#L254). (Probably for most other pytorch dataloaders too)

You can play around with the crop size of 3D MRI images, the downsampling option, the normalization methods etc. here in this file. 

The dataloader now loads 'COBRE','BrainGluSchi' and 'NMorph' three specific datasets among all ten folds. But you can change this too (e.g. load another MCIC dataset).

### vgg.py ###

As mentioned above, if you want to use a 3D vgg model, you need to look at this file 
(and uncomment ['import vgg'](https://github.com/TianYe10/Schiz_Classification_BMEN4460_DLBI/blob/3a8a1d596342136bb5e7cd3570cca7866f2e9ab4/main_10fold.py#L20),['model_names' variable](https://github.com/TianYe10/Schiz_Classification_BMEN4460_DLBI/blob/f39eccb487a5af33298b6d6efda56dad7815886f/main_10fold.py#L32), ['--arch' argument](https://github.com/TianYe10/Schiz_Classification_BMEN4460_DLBI/blob/f39eccb487a5af33298b6d6efda56dad7815886f/main_10fold.py#L54) and [model initiation / definition](https://github.com/TianYe10/Schiz_Classification_BMEN4460_DLBI/blob/f39eccb487a5af33298b6d6efda56dad7815886f/main_10fold.py#L166) in the main file)

The options for different kinds of VGGs include:  'VGG', 'vgg6_bn', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',  'vgg19_bn', 'vgg19'.

If you want to learn more about deep learning (even beyond this class), it is highly recommended to be very familiar with [vgg models](https://paperswithcode.com/method/vgg) and how it is coded up in Python. This code file by itself can be an excellent referable resource.


### command.sh ###

You should be able to open it using any text reader / editor. It gives you an example of running command.

Though it looks long, the major body is just 'python main_10fold.py'. All following words are how you name and define your model saving path and log path (which is also very important, you need to rewrite it to your own laptop path / Google Drive directory), parameters such as batch_size, learning rate, cuda index, etc. (Or you can just remove those unneeded arguments! as long as you modify codes accordingly and it doesn't gives you error)

Now the default setting uses fold 2 to validate, fold 9 to test, and all other eight folds to train. This setting is recommended to be kept like this.

It is a standard command line that has been modified and used for years by multiple lab members.



## Final Comments ##

Please disregard any codes related to 'DeepC' or 'AI-CBV'. We are starting with using only structral T1w MRI for classification. CBV input as the functional map of T1w images is a good next step for novelty in this research project. To know more about DeepC and CBV (cerebral blood volume), please read this good paper in 2020: https://ieeexplore.ieee.org/abstract/document/9098323.

I understand all these codes can be a lot and  a bit overwhelming if you are new to Python or deep learning. Please relax, and do try debugging by yourself and Google solutions if errors pop up - this is how you quickly improve! Then you can discuss with teammates and ask your tutor / TA / professor. As long as you try your best to run codes and push the project, you will be fine!

One final tip: Resources in deep learning includes 'HUMAN' and 'MACHINE'. Not only you need people help your coding or monitor your running models, but also you should grab all machine resources to speed up your project. [Colab](https://research.google.com/colaboratory/faq.html) and [Kaggle](https://www.kaggle.com/code/dansbecker/running-kaggle-kernels-with-a-gpu) are good online gpu resources to start, and there are much more to explore. ***Paid service (monthly membership/upgraded gpu, etc.) are accepted, but not required for our project.*** If you really run out of GPU memory, try make your batch size or input image size smaller. 

(Your project tutor: Ye Tian (yt2793@columbia.edu)
