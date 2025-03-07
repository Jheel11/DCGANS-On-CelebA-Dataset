

---

## **Deep Convolutional Generative Adversarial Network (DCGAN) in PyTorch**  

This repository contains an implementation of **DCGAN (Deep Convolutional Generative Adversarial Network)** in PyTorch. The model is trained on the **CelebA dataset** to generate realistic human face images.  

---

## **1. Dataset: CelebA**  

The **CelebA (Large-Scale CelebFaces Attributes Dataset)** is a widely used dataset for face-related deep learning tasks. It consists of:  
- **202,599** images of celebrity faces  
- Images with varying poses, facial expressions, and occlusions  
- **Aligned and cropped faces**  

We use the **aligned and cropped face images** for training the DCGAN model. The dataset is automatically downloaded via the Kaggle API.  

---

## **2. Dataset Preprocessing Steps**  

### **Step 1: Downloading the Dataset**  
The CelebA dataset is downloaded using the Kaggle API. Make sure you have a valid `kaggle.json` file in your working directory.  

```bash
# Move kaggle.json to the appropriate directory
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download CelebA dataset
!kaggle datasets download -d jessicali9530/celeba-dataset
```

### **Step 2: Extracting the Dataset**  
The dataset is extracted to the `data/` directory.  
```python
import zipfile

with zipfile.ZipFile("celeba-dataset.zip", "r") as zip_ref:
    zip_ref.extractall("data")
```

### **Step 3: Data Transformations**  
Before feeding images into the model, we apply **image transformations** using `torchvision.transforms`:  
- **Resize** the images to `64x64`  
- **CenterCrop** to remove unnecessary background  
- **Convert to Tensor** for deep learning processing  
- **Normalize** pixel values to `[-1, 1]` (for stability in GAN training)  

```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

### **Step 4: Load the Dataset**  
We use `torchvision.datasets.ImageFolder` to load the images into a PyTorch **DataLoader** for efficient batch processing.  
```python
import torchvision
from torch.utils.data import DataLoader

dataset = torchvision.datasets.ImageFolder(root="data/img_align_celeba", transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
```

---

## **3. Training the DCGAN Model**  

### **Run the Training Script**  
To train the model, simply execute the script:  
```bash
python dcgan.py
```

### **Model Architecture**  

#### **Generator**  
The **generator** converts a **random noise vector (latent space)** into a `64x64` RGB image using **transposed convolutions (ConvTranspose2D)**. It includes:  
- **Batch Normalization** (to stabilize training)  
- **ReLU activation** (to introduce non-linearity)  
- **Tanh activation** in the output layer (for pixel normalization)  

#### **Discriminator**  
The **discriminator** is a CNN-based **binary classifier** that differentiates real and fake images using **convolutional layers**. It includes:  
- **LeakyReLU activation** (to prevent dead neurons)  
- **Batch Normalization** (for stability)  
- **Sigmoid activation** in the output layer (for probability score)  

---

## **4. Testing the Model (Generating Images)**  

After training, we can generate new face images using the trained Generator.  

```python
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

# Function to generate and display images
def generate_images(num_images=16):
    generator.eval()
    latent_dim = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate random noise
    noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
    
    # Generate images
    with torch.no_grad():
        generated_images = generator(noise)

    # Display generated images in a grid
    grid = vutils.make_grid(generated_images, nrow=4, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
    plt.show()

# Generate sample images
generate_images()
```

---

## **5. Expected Outputs**  

### **During Training**  
- The script prints the **Discriminator loss (D Loss)** and **Generator loss (G Loss)** for each epoch:  
  ```
  Epoch [1/20] | D Loss: 0.5841 | G Loss: 1.3425
  Epoch [2/20] | D Loss: 0.4962 | G Loss: 1.6783
  ...
  ```
- Loss trends:
  - **D Loss** should stabilize around **0.5** (indicating a balanced Discriminator).
  - **G Loss** should gradually decrease, improving generated image quality.

### **Generated Images**  
- Early epochs: Generated images look noisy and unrealistic.  
- Mid-training: Face-like structures emerge but still blurry.  
- Later epochs: Faces become clearer and more detailed.  


---

## **6. Hyperparameters**  

| Parameter           | Value  |
|---------------------|--------|
| **Image size**     | 64x64  |
| **Batch size**     | 64     |
| **Latent vector size** | 100 |
| **Generator features** | 64 |
| **Discriminator features** | 64 |
| **Learning rate**  | 0.0005 |
| **Optimizer**      | AdamW  |
| **Epochs**         | 20     |

---

## **7. Conclusion**  

- **DCGAN successfully generates realistic human faces** after training on the CelebA dataset.  
- **The quality of generated images improves with training** as the Generator and Discriminator learn better representations.  
- This implementation can be extended to other datasets and fine-tuned for **higher resolution image generation**.  

---

## **8. References**  

- Radford, A., Metz, L., & Chintala, S. (2015). [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434).
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).  

---

