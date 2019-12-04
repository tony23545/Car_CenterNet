import torch
from torch.utils.data import Dataset, DataLoader
from processing import preprocess_image, get_mask_and_regr
import numpy as np

IMG_WIDTH = 1024
IMG_HEIGHT = IMG_WIDTH // 16 * 5
MODEL_SCALE = 8

def preprocess_image(img, flip=False):
    # get the bottom half of the image
    # this is reasonable becasue the upper half is mostly sky or very far away
    img = img[img.shape[0] // 2:]
    bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    bg = bg[:, :img.shape[1] // 6]
    img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    if flip:
        img = img[:, ::-1]
    return (img / 255).astype('float32')

# regression output order should be:
# x_offset, y_offset, z_transformed, yaw, roll, pitch_cos, pitch_sin, pitch_cpositive, pitch_cnegative
def process_pitch(pitch):
    pitch = rotate(pitch, 0) # make it around -pi to pi
    if pitch > 0: # pi / 2 bin
      bin_ = 7
      pitch -= np.pi / 2
    else: # -pi / 2 bin
      bin_ = 8
      pitch += np.pi / 2
    return bin_, pitch

def get_mask_and_regr(img, labels, flip=False):
    mask = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE], dtype='float32')
    regr = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE, 9], dtype='float32')
    coords = str2coords(labels)
    xs, ys = get_img_coords(labels)
    for x, y, regr_dict in zip(xs, ys, coords):
        if flip:
            regr_dict['x'] = -regr_dict['x']
            regr_dict['pitch'] = -regr_dict['pitch']
            regr_dict['roll'] = -regr_dict['roll']
        x, y = y, x
        xf = float(x - img.shape[0] // 2) * IMG_HEIGHT / (img.shape[0] // 2) / MODEL_SCALE # float coord
        x = np.round(xf).astype('int')
        yf = float(y + img.shape[1] // 6) * IMG_WIDTH / (img.shape[1] * 4 / 3) / MODEL_SCALE
        y = np.round(yf).astype('int')
        if x >= 0 and x < IMG_HEIGHT // MODEL_SCALE and y >= 0 and y < IMG_WIDTH // MODEL_SCALE:
            mask[x, y] = 1
            # x, y, z
            regr[x, y, 0] = xf - x # delta x
            regr[x, y, 1] = yf - y # delta y
            regr[x, y, 2] = np.log(1e-5 + regr_dict['z'])

            # yaw, roll
            regr[x, y, 3] = regr_dict['yaw']
            regr[x, y, 4] = regr_dict['roll']

            # pitch
            bin_, delta_pitch = process_pitch(regr_dict['pitch'] - math.atan2(regr_dict['z'], regr_dict['x']))
            regr[x, y, bin_] = 1
            regr[x, y, 5] = cos(delta_pitch)
            regr[x, y, 6] = sin(delta_pitch)
            
    if flip:
        regr[:, :, 1] = -regr[:, :, 1]
        mask = np.array(mask[:, ::-1])
        regr = np.array(regr[:, ::-1])

    return mask, regr

class CarDataset(Dataset):
    """Car dataset."""

    def __init__(self, dataframe, root_dir, training=True, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image name
        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)
        
        flip = False
        if self.training:
            flip = np.random.randint(10) >= 5

        # Read image
        img0 = imread(img_name, True)
        img = preprocess_image(img0, flip=flip)
        img = np.rollaxis(img, 2, 0)
        
        # Get mask and regression maps
        mask, regr = get_mask_and_regr(img0, labels, flip = flip)
        regr = np.rollaxis(regr, 2, 0)
        
        return [img, mask, regr]
