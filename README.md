# DCGANS-On-CelebA-Dataset

## Introduction  

This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** using PyTorch to generate realistic human face images. GANs (Generative Adversarial Networks) are a class of deep learning models consisting of two neural networks—the **Generator** and the **Discriminator**—that compete with each other in a game-theoretic framework to improve the quality of generated data.  

### What is DCGAN?  
DCGAN (Deep Convolutional Generative Adversarial Network) is an improved version of the traditional GAN architecture that specifically uses **convolutional layers** instead of fully connected layers. Introduced by Radford et al. in 2015, DCGANs are designed to generate high-quality images by leveraging **deep convolutional and transposed convolutional layers**.  

- **Generator**: Takes random noise as input and generates realistic images.
- **Discriminator**: Classifies images as real (from dataset) or fake (from Generator).  
- The two networks are trained simultaneously: the Generator improves in creating realistic images, while the Discriminator improves in distinguishing real from fake.  

DCGANs have been widely used for **image synthesis, super-resolution, and style transfer**.  

---

## Dataset: CelebA (Large-Scale CelebFaces Attributes Dataset)  

The **CelebA** dataset is a popular large-scale dataset for facial attribute recognition and generative modeling. It contains:  

- **202,599** celebrity face images  
- **10,177** unique identities  
- Images of varying poses, facial expressions, and occlusions  
- **40 binary attribute labels** (e.g., smiling, male/female, glasses, etc.)  

In this implementation, we use **aligned and cropped** face images from CelebA, resizing them to **64x64 pixels** to train our DCGAN model. The dataset is publicly available on Kaggle and is automatically downloaded via the script.

---

## Dependencies  

Ensure you have the required Python libraries installed before running the code:  

```bash
pip install torch torchvision matplotlib tqdm kaggle
```

---

## Usage  

### 1. Download and Extract Dataset  
The script automatically downloads the **CelebA dataset** from Kaggle and extracts the images. Ensure you have your `kaggle.json` API key configured correctly.

### 2. Train the DCGAN Model  
Run the training script:  
```bash
python dcgan.py
```
This will:  
- Load and preprocess the dataset  
- Initialize the **Generator** and **Discriminator** networks  
- Train the model for **20 epochs**  

### 3. Generate Fake Images  
After training, the model will generate fake images of faces. Generated images are saved in the current directory after each epoch.  

---

## Model Architecture  

### **Generator**  
- Uses **transposed convolutions (ConvTranspose2D)** to upscale noise into an image  
- Uses **Batch Normalization** and **ReLU activations** to stabilize training  
- Outputs a **64x64 RGB image** with values between [-1, 1] (Tanh activation)  

### **Discriminator**  
- Uses **convolutional layers** to extract features from images  
- Uses **LeakyReLU** for better gradient flow  
- Outputs a **single probability value** (real or fake)  

---

## Hyperparameters  

| Parameter           | Value  |
|---------------------|--------|
| Image size         | 64x64  |
| Batch size         | 64     |
| Latent vector size | 100    |
| Generator features | 64     |
| Discriminator features | 64 |
| Learning rate      | 0.0005 |
| Optimizer         | AdamW  |
| Epochs            | 20     |

---

## Results  
At the end of training, the generator creates realistic **human-like faces** from random noise. The quality of generated images improves as training progresses.  

Sample generated images are saved after each epoch and can be viewed using:  
```python
generate_images(num_images=16)
```

---

