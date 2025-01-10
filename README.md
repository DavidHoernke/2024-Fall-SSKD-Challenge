# -2024-Fall-SSKD-Challenge-

## Network Architecture 
The overview of the network architecture the group has built is designed like a U-net-Like 
architecture. This consists of and encoder, bottleneck, decoder and skip connections. The 
encoder is used for down-sampling and feature extraction, bottleneck for capturing high
level representations, decoder for up-sampling and reconstructing the segmentation map, 
and skip connections to preserve spatial information and improve reconstruction. The 
model employs techniques such as depth wise separable convolutions (Howard et al., 
2017), residual connections (He et al., 2016), and a U-Net-like structure (Ronneberger et 
al., 2015) to achieve these goals. This design architecture incorporates advanced 
techniques, including residual depth wise separable convolutions for efficient and reduced 
parameter count, making it lightweight yet effective.

The core components of the model can be broken down into 5 steps:

### Residual Depth wise Separable Convolutions 
Each convolution block combines depth wise and pointwise convolutions. Depth wise 
performs spatial convolutions over each channel independently, while pointwise applies a 
1x1 convolution to combine all the channel’s information 
### Encoder 
The encoder will consist of 3 stages of encoding, each of these will have two residual 
depthwise separable convolution blocks. The encoder reduces the spatial dimensions 
using MaxPooling with kernel size of 2 and stride 2. 
3 Stages: 3 to 64, 64 to 128, 128 to 256 
### Bottleneck 
The bottleneck consists of a pair of residual depthwise separable convolution layers with 
dilated convolutions which are used to expand the receptive field and capture context 
without the need for additional parameters. The dilation rates were used at 2 and 4, the 
bottleneck also includes a dropout (0.5) for regularization. 
### Decoder 
As with the encoder the decoder has 3 steps of decoding which each come with, 
transposed convolutions (used for up-sampling), and residual depth wise separable 
convolutions (used for feature extraction). The Decoder combines up-sampled features 
with corresponding encoder features via skip connections for spatial detail retention. The 
decoder will also reconstruct segmentation maps by reducing the channel dimensions 
from 512 to 256 to 128 to 64.

### Final Output 
The final output will be presented as a 1x1 convolution which generates the segmentation 
map with the number of channels equal to the number of classes. The output is then 
resized to match the input dimensions using bilinear interpolation. 

Some of the key advantages with this design approach are the lightweight design used for 
efficient depth wise separable convolutions which minimize parameter count. Feature 
preservation utilizing skip connections to retain spatial context from the encoder 
Knowledge distillation used to enhance the model’s performance while keeping it 
computationally efficient. Lastly comprehensive validation which combines quantitative 
metrics and qualitative visuals.  

## Utilization of Knowledge Distillation 
Knowledge distillation was implemented to enhance the performance of a student model 
by transferring knowledge from a larger teacher model. The objective of the project was to 
leverage the generalization capabilities of the larger model while maintaining the efficiency 
and computational resources of the student model. The teacher model remained frozen 
during the training to serve as a reliable source of knowledge. The teacher model was a pre
trained FCN-ResNet50 fine-tuned on the PASCAL VOC 2012 dataset. 
Distillation loss was calculated using 2 components hard loss and soft loss.  
Hard loss was the standard entropy loss calculated between the student’s output and the 
ground truth segmentation mask. This ensures the student model will learn directly from 
the labeled data.  
Soft loss was handled using the Kullback-Leibler Divergence to align the student’s 
predicted probability distribution with that of the teacher. The Teacher logits were softened 
using the temperature parameter. This allowed for the student to capture the inter-class 
relationships.  
Total loss is given by: 
![image](https://github.com/user-attachments/assets/fd910262-0cd4-4f52-bf41-c5514cc6b2fc)

## Training Plots:
![image](https://github.com/user-attachments/assets/8b90fb1e-0e66-4c53-9088-c0b2c756289c)

Howard, A. G., et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile 
Vision Applications." arXiv preprint arXiv:1704.04861, 2017. (Depthwise Separable 
Convolutions) 
He, K., et al. "Deep Residual Learning for Image Recognition." CVPR, 2016. (Residual 
Connections) 
Ronneberger, O., et al. "U-Net: Convolutional Networks for Biomedical Image 
Segmentation." MICCAI, 2015. (U-Net Structure) 
