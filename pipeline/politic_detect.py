"""
created by: Donghyeon Won
"""

from __future__ import print_function
import os
import argparse
import numpy as np
import pandas as pd
import time
import shutil
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.models as models

from protestDetection.util import ProtestDatasetEval, modified_resnet50


def eval_one_dir(img_dir, model, rate):
        """
        return model output of all the images in a directory
        """
        model.eval()
        # make dataloader
        dataset = ProtestDatasetEval(img_dir = img_dir, rate=rate)
        data_loader = DataLoader(dataset,
                                num_workers = 4,
                                batch_size = 16)
        # load model

        outputs = []
        imgpaths = []

        n_imgs = len(dataset)
        with tqdm(total=n_imgs) as pbar:
            for i, sample in enumerate(data_loader):
                imgpath, input = sample['imgpath'], sample['image']
                if torch.cuda.is_available():
                    input = input.cuda()

                input_var = Variable(input)
                output = model(input_var)
                outputs.append(output.cpu().data.numpy())
                imgpaths += imgpath
                if i < n_imgs / 16:
                    pbar.update(16)
                else:
                    pbar.update(n_imgs%16)

        predictions = np.concatenate(outputs)[:,0]
        elapsed_time = []
        for i in range(n_imgs):
            elapsed_time.append(i*rate/24)
        elapsed_time = np.array(elapsed_time) 
        return predictions, elapsed_time

def infer(img_dir, model_path, rate=4):

    # load trained model
    model = modified_resnet50()
    if torch.cuda.is_available():
        model = model.cuda()
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f)['state_dict'])
    print("*** calculating the model output of the images in {img_dir}"
            .format(img_dir = img_dir))

    # calculate output
    pred = eval_one_dir(img_dir, model, rate=rate)
    
    return pred