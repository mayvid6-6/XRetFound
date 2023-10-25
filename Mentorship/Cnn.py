import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf #Machine Learning Library

from tensorflow.keras.applications.vgg16 import VGG16 as Model #VGG16 is a pretrained CNN that has 16 layers, it can classify images into 1000+ categories

model = Model(weights='imagenet', include_top=True) #Creates an instance of the imagenet model, include_top is whether the final layers will be dense or not
# model.summary()


from tensorflow.keras.preprocessing.image import load_img #Loadimg lets you load images in a specific format from local files
from tensorflow.keras.applications.vgg16 import preprocess_input #Preprocess images to be used with the vgg16 model

# Image titles
image_titles = ['Husky'] # Headings for each image

# Load images and Convert them to a Numpy array
img1  = load_img('Mentorship/images/husky.jpg', target_size=(224, 224))
images = np.asarray(np.array(img1)) #Makes array from Images

# Preparing input data for VGG16
X = preprocess_input(images)

# Rendering the imgs in a 1x3 format
plt.imshow(img1)
plt.tight_layout() #minimzes window size
plt.show()


def model_modifier_function(cloned_model):
    cloned_model.layers[-1].activation = tf.keras.activations.linear #replaces all the activation functions of the last layer to linear
    
    
from tf_keras_vis.utils.scores import CategoricalScore

# 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
score = CategoricalScore([249]) #collects the scores from model output which is for categorical classification


from matplotlib import cm  #color map
from tf_keras_vis.gradcam import Gradcam #Gradcam

# Create Gradcam object
gradcam = Gradcam(model, model_modifier=model_modifier_function(model), clone=True) #creates an instance of gradcam object with model modifier being replace2linear

# Generate heatmap with GradCAM
cams = [gradcam(score, X, penultimate_layer=2), gradcam(score, X, penultimate_layer=3), gradcam(score, X, penultimate_layer=4), gradcam(score, X, penultimate_layer=5), gradcam(score, X, penultimate_layer=6), gradcam(score, X, penultimate_layer=7), gradcam(score, X, penultimate_layer=8), gradcam(score, X, penultimate_layer=9), gradcam(score, X, penultimate_layer=10), gradcam(score, X, penultimate_layer=11), gradcam(score, X, penultimate_layer=12), gradcam(score, X, penultimate_layer=13), gradcam(score, X, penultimate_layer=14), gradcam(score, X, penultimate_layer=15), gradcam(score, X, penultimate_layer=16), gradcam(score, X, penultimate_layer=17)]
cam = gradcam(score, X, penultimate_layer=-1) #creates gradcam heatmap using specified layer
#could create a subplot that shows the heatmap through the layers
# Renders the heat map with the image


# f, ax = plt.subplots(nrows=1, ncols=16, figsize=(16, 16))
# for i in range(16):
#     for j in range(len(cams[i])):
#         heatmap = np.uint8(cm.jet(cams[i][j])[..., :3] * 255)
#         ax[i].set_title(str(i), fontsize=2)
#         ax[i].imshow(img1)
#         ax[i].imshow(heatmap, cmap='jet', alpha=0.5) # overlay
# plt.tight_layout()
# plt.show()

f, ax = plt.subplots(nrows=4, ncols=4, figsize=(16, 16))
for i in range(4):
    for j in range(4):
        for x in range(len(cams[i*4 + j])):
            heatmap = np.uint8(cm.jet(cams[i*4 + j][x])[..., :3] * 255)
            ax[i][j].set_title(str(i*4 + j), fontsize=8)
            ax[i][j].imshow(img1)
            ax[i][j].imshow(heatmap, cmap='jet', alpha=0.5) # overlay
            ax[i][j].axis("off")
plt.tight_layout()
plt.show()