from torch.utils.data import DataLoader
import torch
import numpy as np
from HyperVD.model import Model
from HyperVD.dataset import Dataset
from HyperVD.test import test
import HyperVD.option as option
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def infer():
    print('perform testing...')
    args = option.parser.parse_args()
    args.device = 'cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu'
    # device = torch.device("cuda")

    test_loader = DataLoader(Dataset(args, 
                                     '/home/www/data/data/saigonmusic/Dev_AI/kiendn/movie_classify/features/list/rgb.list',
                                     '/home/www/data/data/saigonmusic/Dev_AI/kiendn/movie_classify/features/list/audio.list',
                                     test_mode=True),
                              batch_size=5, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    model = Model(args)
    model = model.to(args.device)
    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('./HyperVD/ckpt/pretrained.pkl', map_location=torch.device('cpu')).items()})

    pred = test(test_loader, model, args)
    return pred

if __name__ == "__main__":
    print(infer())