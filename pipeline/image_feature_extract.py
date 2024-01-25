import torch
from torch.autograd import Variable

from torchvision import transforms
import numpy as np

import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pytorchi3d.videotransforms as videotransforms
from pytorchi3d.pytorch_i3d import InceptionI3d
from pytorchi3d.charades_dataset_full import Charades as Dataset, video_to_tensor

class ImageFeatureExtractor:
    def __init__(
        self,
        root="./",
        mode="rgb",
        isTrain=False,
        batch_size=1,
        save_dir="./features/img_features",
        load_model="/home/www/data/data/saigonmusic/Dev_AI/kiendn/checkpoint/models/rgb_imagenet.pt",
    ):
        self.root = root
        self.mode = mode
        self.isTrain = isTrain
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.load_model = load_model

    def extract_image_features(self):
        first_transforms = transforms.Compose([videotransforms.FirstCrop(224)])
        second_transforms = transforms.Compose([videotransforms.SecondCrop(224)])
        third_transforms = transforms.Compose([videotransforms.ThirdCrop(224)])
        fourth_transforms = transforms.Compose([videotransforms.FourthCrop(224)])
        center_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

        dataset = Dataset(self.root, self.mode, True, num=-1, save_dir=self.save_dir)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)

        transformers = {
               '0': center_transforms,
               '1': first_transforms,
               '2': second_transforms,
               '4': third_transforms,
               '3': fourth_transforms,
                }

        i3d = InceptionI3d(400, in_channels=3)
        i3d.replace_logits(400)
        i3d.load_state_dict(torch.load(self.load_model))
        i3d.cuda()
        i3d.train(False)  # Set model to evaluate mode

        # Iterate over data.
        saved_list = []
        elapsed_frames = []

        for data in dataloader:
            inputs, _, name = data
            if name == 0:
                continue
            inputs = inputs.numpy()
            for phase in ['0', '1', '2', '3', '4']:
                # get the inputs
                crop_inputs = transformers[phase](inputs[0])
                crop_inputs = video_to_tensor(crop_inputs).unsqueeze(0)
                b,c,t,h,w = crop_inputs.shape
                features = []
                for start in range(0, t, 16): 
                    end = start + 16
                    ip = Variable(torch.from_numpy(crop_inputs.numpy()[:,:,start:end]).cuda())
                    if ip.shape[2] != 16:
                        continue
                    if phase == '0':
                        elapsed_frames.append([start, end-1]) 
                    features.append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())      
                print(os.path.join(self.save_dir, name[0]+f"__{phase}"))
                np.save(os.path.join(self.save_dir, name[0]+f"__{phase}"), np.concatenate(features, axis=0).reshape(-1, 1024))
                saved_list.append(f"{os.path.join(self.save_dir, name[0]+'__'+phase)}.npy")
        return saved_list, elapsed_frames