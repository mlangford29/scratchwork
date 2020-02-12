# Michael Langford
# 1/4/20
# simple script for superresolution

import numpy as np
from PIL import Image
from ISR.models import RDN
import scipy.misc
import os


rdn = RDN(arch_params={'C':3, 'D':10, 'G':64, 'G0':64, 'x':2})
rdn.model.load_weights('weights/rdn-C3-D10-G64-G064-x2_PSNR_epoch134.hdf5')

# just going to name them by the order they were given in ugh
i = 0

# loop through all the files in the images folder
for file in os.listdir('data/input/images'):

	if file == '.DS_Store':
		continue
	
	i += 1
	img = Image.open(os.path.join('data/input/images', file))
	lr_img = np.array(img)

	print('Predicting on {}'.format(file))
	sr_img = rdn.predict(lr_img, by_patch_of_size=2)

	scipy.misc.imsave('results/{}.png'.format(i), sr_img)

#Image.fromarray(sr_img)