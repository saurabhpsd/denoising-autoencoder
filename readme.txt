# Image Denoising Autoencoder

This project implements an image denoising autoencoder using Convolutional Neural Networks (CNN) in TensorFlow/Keras. The model learns to remove noise from color images by training on noisy and clean image pairs.

---

## Features

- Loads color images from a specified folder.
- Adds synthetic Gaussian noise to the images.
- Trains a convolutional autoencoder to denoise images.
- Visualizes original, noisy, and denoised images side-by-side.
- Supports customizable image size and batch size.

---

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

You can install the required packages using:

```bash
pip install tensorflow numpy matplotlib

Setup and Usage
Prepare your dataset:

Place your clean images (JPEG, PNG, etc.) inside a folder, e.g.,
C:\Users\saura\Desktop\denoising_autoencoder\images

The images will be resized to 128x128 pixels by default.

Run the script:

Update the image_folder path in the Python script to point to your images folder.

Run the script:

bash
Copy
Edit
python denoising_autoencoder.py
Output:

The model trains for 10 epochs (default).

After training, it shows a visualization of noisy, denoised, and original images.

How It Works
Data loading: Reads images from the folder and normalizes pixel values.

Noise addition: Adds Gaussian noise to simulate noisy images.

Autoencoder architecture: CNN-based encoder-decoder network.

Training: Learns to reconstruct clean images from noisy inputs.

Visualization: Displays comparison images to evaluate denoising quality.

Customization
Change target_size in load_images_from_folder() for different input sizes.

Adjust noise_factor to control noise intensity.

Modify training parameters (epochs, batch size) in autoencoder.fit().

References
Based on standard convolutional autoencoder architecture for image denoising.

TensorFlow/Keras official documentation.

🧠 Architecture
Encoder: Conv2D + MaxPooling + BatchNorm layers
Latent Space: Bottleneck with compressed representation
Decoder: Conv2D + UpSampling + BatchNorm layers
Loss Function: Binary Crossentropy
Optimizer: Adam

📊 Evaluation Metrics
Metric	Average
PSNR	22.38 dB
SSIM	0.7984

💡 Applications
🏥 Medical Imaging – denoise MRI, CT, ultrasound images
🎥 Surveillance & Security – clean noisy video frames
📷 Photography – enhance low-light smartphone images
🛰️ Remote Sensing – improve clarity of satellite data
🚗 Autonomous Vehicles – preprocess sensor inputs
🔬 Scientific Imaging – denoise microscopy or astronomy data

🔮 Future Enhancements
Handle multiple noise types (salt-and-pepper, speckle)
Train on real noisy datasets
Optimize for real-time video streams
Expand to high-res image support
Add GUI using Tkinter/Flask
Integrate advanced models like U-Net or GANs
