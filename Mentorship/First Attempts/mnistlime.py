import sys
import numpy as np #Math Library
import matplotlib.pyplot as plt #Allows you to display plots
from skimage.color import gray2rgb, rgb2gray, label2rgb #gray2rgb creats an RGB representation of a gray-level image, label2rgb returns an rgb image where color coded labels are painted over the image, rest are self explanatory
from sklearn.datasets import fetch_openml #Loads datasets
mnist = fetch_openml('mnist_784', parser='auto') #mnist_784 is the name of the dataset and it is assigned to variable mnist

#For lime_image to work correctly, we need to turn each image into a color image
#That is where we use gray2rgb function
data = mnist.data.to_numpy()
data = np.reshape(data, (-1, 28, 28))
X_vec = np.stack([gray2rgb(iimg) for iimg in data], 0).astype(np.uint8) #converts every pixel in the image to rgb using list comprehension
y_vec = mnist.target.astype(np.uint8)

#%matplotlib inline #used for notebook
fig, ax1 = plt.subplots(1, 1) #Creates a plot
ax1.imshow(X_vec[0], interpolation = 'none')
ax1.set_title('Digit: {}'.format(y_vec[0])) #sets the title for the first image

plt.show()



from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Normalizer

class PipeStep(object):
    """
    Wrapper for turning functions into pipeline transforms (no-fitting)
    """
    def __init__(self, step_func):
        self._step_func=step_func
    def fit(self,*args):
        return self
    def transform(self,X):
        return self._step_func(X)


makegray_step = PipeStep(lambda img_list: [rgb2gray(img) for img in img_list])
flatten_step = PipeStep(lambda img_list: [img.ravel() for img in img_list])

simple_rf_pipeline = Pipeline([
    ('Make Gray', makegray_step),
    ('Flatten Image', flatten_step),
    #('Normalize', Normalizer()),
    #('PCA', PCA(16)),
    ('RF', RandomForestClassifier())
                              ])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec, train_size=0.55)

simple_rf_pipeline.fit(X_train, y_train)


import os,sys
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
    
    
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
explainer = lime_image.LimeImageExplainer(verbose = False)
segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)

explanation = explainer.explain_instance(X_test[0], 
                                         classifier_fn = simple_rf_pipeline.predict_proba, 
                                         top_labels=10, hide_color=0, num_samples=10000, segmentation_fn=segmenter)

temp, mask = explanation.get_image_and_mask(y_test[0], positive_only=True, num_features=10, hide_rest=False, min_weight = 0.01)
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (8, 4))
ax1.imshow(label2rgb(mask,temp, bg_label = 0), interpolation = 'nearest')
ax1.set_title('Positive Regions for {}'.format(y_test[0]))
temp, mask = explanation.get_image_and_mask(y_test[0], positive_only=False, num_features=10, hide_rest=False, min_weight = 0.01)
ax2.imshow(label2rgb(3-mask,temp, bg_label = 0), interpolation = 'nearest')
ax2.set_title('Positive/Negative Regions for {}'.format(y_test[0]))

plt.show()

fig, m_axs = plt.subplots(2,5, figsize = (12,6))
for i, c_ax in enumerate(m_axs.flatten()):
    temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=1000, hide_rest=False, min_weight = 0.01 )
    c_ax.imshow(label2rgb(mask,X_test[0], bg_label = 0), interpolation = 'nearest')
    c_ax.set_title('Positive for {}\nActual {}'.format(i, y_test[0]))
    c_ax.axis('off')
    
pipe_pred_test = simple_rf_pipeline.predict(X_test)

print(y_test[0])
print("pipe_pred_test")
print(type(pipe_pred_test))
print(pipe_pred_test)
wrong_idx = np.random.choice(pipe_pred_test)
while wrong_idx not in y_test and wrong_idx not in pipe_pred_test:
    wrong_idx = np.random.choice(pipe_pred_test) #problems on this line

print(wrong_idx)

print(y_test[wrong_idx])

print(pipe_pred_test[wrong_idx])


print('Using #{} where the label was {} and the pipeline predicted {}'.format(wrong_idx, y_test[wrong_idx], pipe_pred_test[wrong_idx]))