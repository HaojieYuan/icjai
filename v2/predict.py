from model import inceptionv4,inceptionresnetv2
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import os
from PIL import Image
import argparse

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = inceptionresnetv2()
    model.load_state_dict(torch.load('IRnet_pretrained.pkl'))
    model = model.to(device)
    
    model.eval()
    # with csv file
    image_names_ = os.listdir(args.input_dir)
    image_paths = []
    
    # without csv file
    image_names = []
    for image_name in image_names_:
        if not image_name.endswith(".csv"):
            image_names.append(image_name)
    
    image_paths = [os.path.join(args.input_dir, image_name) for image_name in image_names]
    imgs = [cv2.resize(cv2.imread(img_path), (299,299)) for img_path in image_paths]

    batch_size = 16
    nb_batch = len(imgs)//batch_size
    rest = len(imgs)%batch_size

    output_labels = []
    to_tensor = transforms.ToTensor()


    for i in range(nb_batch):
        stack_tensors = []
        for image in imgs[i*batch_size:(i+1)*batch_size]:
            image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image = Image.fromarray(image)
            image = to_tensor(image)
            stack_tensors.append(image)

        img_tensors = torch.stack(stack_tensors)
        with torch.no_grad():
            output = model(img_tensors.to(device, dtype=torch.float))
            _, predicted = torch.max(output.data, 1)

        for label in predicted:
            output_labels.append(label)

    rest_batch = imgs[-batch_size:]
    stack_tensors = []
    for image in rest_batch:
        image = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = Image.fromarray(image)
        image = to_tensor(image)
        stack_tensors.append(image)

    img_tensors = torch.stack(stack_tensors)
    with torch.no_grad():
        output = model(img_tensors.to(device, dtype=torch.float))
        _, predicted = torch.max(output.data, 1)

    for label in predicted[-rest:]:
        output_labels.append(label)

    with open(args.output_file, 'w') as writer:
        for img_name, predict_label in zip(image_names, output_labels):
            writer.write('%s,%d\n'%(img_name, predict_label))

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="normal model.")
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_file', type=str)
    main(parser.parse_args())
    
    