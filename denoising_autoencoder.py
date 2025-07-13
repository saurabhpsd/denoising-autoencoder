import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ------------- Step 1: Load and preprocess images -------------

def load_images_from_folder(folder, target_size=(128, 128)):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = load_img(img_path, target_size=target_size)  # Load RGB image and resize
        img = img_to_array(img) / 255.0  # Normalize pixel values to [0,1]
        images.append(img)
    return np.array(images)

# Update this path to your local folder
image_folder = r'C:\Users\saura\Documents\denoising_autoencoder\images\CT_Covid-19'
clean_images = load_images_from_folder(image_folder, target_size=(128, 128))

print(f"Loaded {len(clean_images)} images of shape {clean_images[0].shape}")

# ------------- Step 2: Add noise -------------

noise_factor = 0.5
noisy_images = clean_images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=clean_images.shape)
noisy_images = np.clip(noisy_images, 0., 1.)

# ------------- Step 3: Build the Autoencoder model -------------

input_img = Input(shape=(128, 128, 3))  # Input shape for color images

# Encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # Output with 3 channels (RGB)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

# ------------- Step 4: Train the model -------------

autoencoder.fit(noisy_images, clean_images,
                epochs=3,
                batch_size=8,
                shuffle=True,
                validation_split=0.1)  # 10% data for validation

# ------------- Step 5: Predict on noisy images -------------

denoised_images = autoencoder.predict(noisy_images)

# ------------- Step 6: Visualize some results -------------

n = min(5, len(clean_images))  # Show up to 5 images
plt.figure(figsize=(15, 6))
for i in range(n):
    # Noisy image
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(noisy_images[i])
    plt.title("Noisy")
    plt.axis('off')
    
    # Denoised image
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(denoised_images[i])
    plt.title("Denoised")
    plt.axis('off')
    
    # Original clean image
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(clean_images[i])
    plt.title("Original")
    plt.axis('off')

plt.show()
