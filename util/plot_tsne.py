from tsne import bh_sne
import numpy as np
from skimage.transform import resize
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.image as mpimg

import warnings
warnings.filterwarnings("ignore")

def gray_to_color(img):
    if len(img.shape) == 2:
        img = np.dstack((img, img, img))
    return img

def min_resize(img, size):
    """
    Resize an image so that it is size along the minimum spatial dimension.
    """
    w, h = map(float, img.shape[:2])
    if min([w, h]) != size:
        if w <= h:
            img = resize(img, (int(round((h/w)*size)), int(size)))
        else:
            img = resize(img, (int(size), int(round((w/h)*size))))
    return img

def fit(features):
    features = np.copy(features).astype('float64')
    model = TSNE(n_components=2, n_iter=1000, perplexity=10.0)
    f2d = model.fit_transform(features) 
    return f2d

def image_scatter(features, images, img_res, filenames, res=15000, cval=1.):
    """
    Embeds images via tsne into a scatter plot.

    Parameters
    ---------
    features: numpy array
        Features to visualize

    images: list or numpy array
        Corresponding images to features. Expects float images from (0,1).

    img_res: float or int
        Resolution to embed images at

    res: float or int
        Size of embedding image in pixels

    cval: float or numpy array
        Background color value

    Returns
    ------
    canvas: numpy array
        Image of visualization
    """
    f2d = fit(features)
    images = [gray_to_color(image) for image in images]
    images = [min_resize(image, img_res) for image in images]
    max_width = max([image.shape[0] for image in images])
    max_height = max([image.shape[1] for image in images])

    xx = f2d[:, 0]
    yy = f2d[:, 1]
    x_min, x_max = xx.min(), xx.max()
    y_min, y_max = yy.min(), yy.max()
    # Fix the ratios
    sx = (x_max-x_min)
    sy = (y_max-y_min)
    if sx > sy:
        res_x = sx/float(sy)*res
        res_y = res
    else:
        res_x = res
        res_y = sy/float(sx)*res

    canvas = np.ones((res_x+max_width, res_y+max_height, 3))*cval
    x_coords = np.linspace(x_min, x_max, res_x)
    y_coords = np.linspace(y_min, y_max, res_y)
    for x, y, image, name in zip(xx, yy, images, filenames):
        w, h = image.shape[:2]
        x_idx = np.argmin((x - x_coords)**2)
        y_idx = np.argmin((y - y_coords)**2)
        if name.endswith("8238.jpg"):
          print("%s is at (%d, %d)" % (name, y_idx, x_idx))
        canvas[x_idx:x_idx+w, y_idx:y_idx+h] = image
    plt.figure(figsize=(10,8))    
    plt.imshow(canvas)
    plt.show()
    raw_input()

def ab_plotter(xcoords, ycoords, images, labels):

    ax = plt.subplot(111)
    ax.set_xlim([-30, 30])
    ax.set_ylim([-30, 30])
    
    for x, y, i, l in zip(xcoords, ycoords, images, labels):
        arr_hand = i
        imagebox = OffsetImage(arr_hand, zoom=.1)
        xy = [x, y]               # coordinates to position this image
        
        ab = AnnotationBbox(imagebox, xy,
            xybox=(10., -10.),
            xycoords='data',
            boxcoords="offset points",
            pad=0.0)                                  
        ax.annotate(ab, xy = xy)
    
    # rest is just standard matplotlib boilerplate
    ax.grid(True)
    plt.show()

if __name__ == "__main__":
    x_data = np.loadtxt("eng256-initial_hidden_features", delimiter=",")
    #x_data = np.loadtxt("eng256-ger256-multilingual_initial_hidden_features", delimiter=",")
    y_data = np.loadtxt("val_images", dtype=str)
    y_data = [x.replace("\n","") for x in y_data]
    images = []
    for y in y_data:
      images.append(mpimg.imread(y))
    canvas_mono = image_scatter(x_data, images, 200, y_data, res=15000)
    #canvas_multi = image_scatter(x_data2, images, 100, y_data)

    #coords = fit(x_data)
    #ab_plotter(coords[:,0], coords[:,1], images, y_data)
    
