import numpy as np

from chainercv import transforms
from skimage import transform as skimage_transform


def cv_rotate(img, angle):
    img = img.transpose(1, 2, 0) / 255.
    img = skimage_transform.rotate(img, angle, mode='edge')
    img = img.transpose(2, 0, 1) * 255.
    img = img.astype(np.float32)

    return img


"""
def random_erase(img):
    if np.random.rand(1) < 0.5:    
        side_y = img.shape[1]
        side_x = img.shape[2]
        size = side_x * side_y 
           
        while True:
            S_e = size * np.random.uniform(low=0.02, high=0.3)
            r_e = np.random.uniform(low=0.3, high=1./0.3)

            Y_e = np.sqrt(S_e * r_e)
            X_e = np.sqrt(S_e / r_e)

            x_e = np.random.randint(0, side_x)
            y_e = np.random.randint(0, side_y)


            if x_e + X_e <= side_x and y_e + Y_e <= side_y:
                height = range(0, int(Y_e))
                width = range(0, int(X_e))

                combinations = [(i, j) for i in height for j in width]
                
                for i, j in combinations:
                    r = np.random.uniform(0, 1)
                    #print(i, j)
                    #print(r)
                    #import pdb
                    #pdb.set_trace()
                    img[:, y_e:int(y_e + i), x_e:int(x_e + j)] = r

                #print("return img")
                return img
    else:
        return img
""" 


def cut_out(img):
    if np.random.rand(1) < 0.5:    
        side_y = img.shape[1]
        side_x = img.shape[2]
        size = side_x * side_y 
           
        while True:
            S_e = size * np.random.uniform(low=0.02, high=0.3)
            r_e = np.random.uniform(low=0.3, high=1./0.3)

            Y_e = np.sqrt(S_e * r_e)
            X_e = np.sqrt(S_e / r_e)

            x_e = np.random.randint(0, side_x)
            y_e = np.random.randint(0, side_y)

            if x_e + X_e <= side_x and y_e + Y_e <= side_y:
                img[:, y_e:int(y_e + Y_e + 1), x_e:int(x_e + X_e + 1)] = np.random.uniform(0, 1)

                return img
    else:
        return img


def transform(
        inputs, mean, std, random_angle=15., expand_ratio=1.0,
        crop_size=(32, 32), train=True):
    img, label = inputs
    img = img.copy()

    # Random rotate
    if random_angle != 0:
        angle = np.random.uniform(-random_angle, random_angle)
        img = cv_rotate(img, angle)

    # Cut out
    img = cut_out(img)

    # Standardization
    img -= mean[:, None, None]
    img /= std[:, None, None]

    if train:
        # Random flip
        img = transforms.random_flip(img, x_random=True)
        # Random expand
        if expand_ratio > 1:
            img = transforms.random_expand(img, max_ratio=expand_ratio)
        # Random crop
        if tuple(crop_size) != (32, 32):
            img = transforms.random_crop(img, tuple(crop_size))

    return img, label
