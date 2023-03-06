## P2-Image-Denoising-using-Neural-Networks

We used in this project the _SIDD (Smartphone Image Denoising Dataset)_ dataset which contains 160 pairs GT - Noisy (camera noise), high resolution images. 
The images were splitted in patches due their resolution and high number of parameters that had the used Neural Networks. Have been used 2 Neural networks:
- [DnCNN](https://arxiv.org/pdf/1608.03981.pdf)
- CAE (Convolutional AutoEncoder)
for the denoising task. 

Data pipeline is shown in the following figure:

![pipeline_process](https://user-images.githubusercontent.com/65508171/223204826-4d75dabb-e4cc-4766-b0e8-5a744182dce4.png)

For more information, we wrote a [paper](https://github.com/banarutz/P2-Image-Denoising-using-Neural-Networks/files/10901922/Image_denoising_based_on_Neural_Networks.3._compressed.pdf).
