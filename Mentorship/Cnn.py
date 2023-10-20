import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf #Machine Learning Library

from tensorflow.keras.applications.vgg16 import VGG16 as Model #VGG16 is a pretrained CNN that has 16 layers, it can classify images into 1000+ categories

model = Model(weights='imagenet', include_top=True) #Creates an instance of the imagenet model, include_top is whether the final layers will be dense or not
# model.summary()


from tensorflow.keras.preprocessing.image import load_img #Loadimg lets you load images in a specific format from local files
from tensorflow.keras.applications.vgg16 import preprocess_input #Preprocess images to be used with the vgg16 model

# Image titles
image_titles = ['Goldfish', 'Bear', 'Assault rifle', 'Husky'] # Headings for each image

# Load images and Convert them to a Numpy array
img1, img2, img3, img4  = load_img('Mentorship/images/goldfish.jpg', target_size=(224, 224)), load_img('Mentorship/images/bear.jpg', target_size=(224, 224)), load_img('Mentorship/images/soldiers.jpg', target_size=(224, 224)), load_img('Mentorship/images/husky.jpg', target_size=(224, 224))
images = np.asarray([np.array(img1), np.array(img2), np.array(img3), np.array(img4)]) #Makes array from Images

# Preparing input data for VGG16
X = preprocess_input(images)

# Rendering the imgs in a 1x3 format
f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
for i, title in enumerate(image_titles):
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
plt.tight_layout() #minimzes window size
plt.show()


def model_modifier_function(cloned_model):
    cloned_model.layers[-1].activation = tf.keras.activations.linear #replaces all the activation functions of the last layer to linear
    
    
from tf_keras_vis.utils.scores import CategoricalScore

# 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
score = CategoricalScore([1, 294, 413, 250]) #collects the scores from model output which is for categorical classification


from matplotlib import cm  #color map
from tf_keras_vis.gradcam import Gradcam #Gradcam

# Create Gradcam object
gradcam = Gradcam(model, model_modifier=model_modifier_function(model), clone=True) #creates an instance of gradcam object with model modifier being replace2linear

# Generate heatmap with GradCAM
cam = gradcam(score, X, penultimate_layer=-1) #creates gradcam heatmap using specified layer

# Renders the heat map with the image
f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
for i, title in enumerate(image_titles):
    heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    ax[i].set_title(title, fontsize=16)
    ax[i].imshow(images[i])
    ax[i].imshow(heatmap, cmap='jet', alpha=0.5) # overlay
plt.tight_layout()
plt.show()