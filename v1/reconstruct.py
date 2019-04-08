import argparse
import scipy as sp
import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import os
import pickle
from hashlib import md5
from time import time
import cv2


def reconstruct_image(test_image, dico, V, patch_size = (7, 7), height=299, width=299):
    data = extract_patches_2d(test_image, patch_size)
    data = data.reshape(data.shape[0], -1)
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0)
    dico.set_params(transform_algorithm='omp', transform_n_nonzero_coefs=1)
    code = dico.transform(data)
    patches = np.dot(code, V)
    patches = patches.reshape(len(data), *patch_size)
    reconstruction = reconstruct_from_patches_2d(patches, (height, width))
    return reconstruction


def main(args):
    target_path = './tmp'
    #start_idx = int(args.start_idx)
    #end_idx = int(args.end_idx)
    
    pickle_in = open("./dict.pickle","rb")
    dico = pickle.load(pickle_in)
    pickle_in.close()
    pickle_in = open("./V.pickle","rb")
    V = pickle.load(pickle_in)
    pickle_in.close()
    
    image_names = os.listdir(args.input_dir)
    image_names.sort()
    
    image_paths = []
    image_names_ = []
    for image_name in image_names:
        if not image_name.endswith(".csv"):
            image_paths.append(os.path.join(args.input_dir, image_name))
            image_names_.append(image_name)

    total_num = len(image_names_)
    num_per_worker = total_num//8
    start_idx = num_per_worker*int(args.id)
    if int(args.id) == 7:
        end_idx = total_num
    else:
        end_idx = start_idx + num_per_worker
    #print(start_idx, end_idx)
    
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
        except:
            pass
        
    for image_path, image_name in zip(image_paths[start_idx:end_idx], image_names_[start_idx:end_idx]):
        try:
            img = np.array(cv2.resize(cv2.imread(image_path, 0), (299,299)))
        except:
            #print(image_path)
            continue
        img = img/255.0
        try:
            img = reconstruct_image(img, dico, V)
        except:
            continue
        np.save(os.path.join(target_path, image_name), img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="reconstruct images using pre learned dict.")
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--id', type=str)
    main(parser.parse_args())
