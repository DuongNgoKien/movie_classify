import torch
from torch.autograd import Variable

from torchvision import transforms
import numpy as np

import os, sys, shutil, glob

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import pytorchi3d.videotransforms as videotransforms
from pytorchi3d.pytorch_i3d import InceptionI3d
from pytorchi3d.charades_dataset_full import Charades as Dataset, video_to_tensor, load_rgb_frames

class ImageFeatureExtractor:
    def __init__(
        self,
        list_img_dir="./",
        mode="rgb",
        isTrain=False,
        batch_size=1,
        save_dir="./features/img_features",
        load_model="/home/www/data/data/saigonmusic/Dev_AI/kiendn/checkpoint/models/rgb_imagenet.pt",
        num_ignore_frame=0
    ):
        self.list_img_dir = list_img_dir
        self.mode = mode
        self.isTrain = isTrain
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.load_model = load_model
        self.num_ignore_frame = num_ignore_frame

    def extract_image_features(self):
        first_transforms = transforms.Compose([videotransforms.FirstCrop(224)])
        second_transforms = transforms.Compose([videotransforms.SecondCrop(224)])
        third_transforms = transforms.Compose([videotransforms.ThirdCrop(224)])
        fourth_transforms = transforms.Compose([videotransforms.FourthCrop(224)])
        center_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

        dataset = Dataset(self.list_img_dir, self.mode, False, num_ignore_frame=self.num_ignore_frame, save_dir=self.save_dir)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=1, pin_memory=True)

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
            root, _, name, nf = data
            root, name = root[0], name[0]
            nf = nf.item()
            remain_nf = nf
            if root == 0:
                saved_list.append(os.path.join(self.save_dir, name+'__0.npy'))
                saved_list.append(os.path.join(self.save_dir, name+'__1.npy'))
                saved_list.append(os.path.join(self.save_dir, name+'__2.npy'))
                saved_list.append(os.path.join(self.save_dir, name+'__3.npy'))
                saved_list.append(os.path.join(self.save_dir, name+'__4.npy'))
                continue
            f_start, f_end = 1, 0
            features = {'0':[],'1':[], '2':[], '3':[], '4':[]}
            while remain_nf > 3040:
                f_end = f_end + 3040
                inputs = load_rgb_frames(root, name, f_start, f_end)
                remain_nf -= 3040
                for phase in ['0', '1', '2', '3', '4']:
                    # get the inputs
                    crop_inputs = transformers[phase](inputs)
                    crop_inputs = video_to_tensor(crop_inputs).unsqueeze(0)
                    b,c,t,h,w = crop_inputs.shape
                    for start in range(0, t, 16): 
                        end = start + 16
                        ip = Variable(torch.from_numpy(crop_inputs.numpy()[:,:,start:end]).cuda())
                        if ip.shape[2] != 16:
                            continue
                        if phase == '0':
                            elapsed_frames.append([f_start+start, f_start+end-1]) 
                        features[phase].append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())
                f_start = f_start + 3040

            inputs = load_rgb_frames(root, name, f_start, nf)
            for phase in ['0', '1', '2', '3', '4']:
                # get the inputs
                crop_inputs = transformers[phase](inputs)
                crop_inputs = video_to_tensor(crop_inputs).unsqueeze(0)
                b,c,t,h,w = crop_inputs.shape
                for start in range(0, t, 16): 
                    end = start + 16
                    ip = Variable(torch.from_numpy(crop_inputs.numpy()[:,:,start:end]).cuda())
                    if ip.shape[2] != 16:
                        continue
                    if phase == '0':
                        elapsed_frames.append([f_start+start, f_start+end-1]) 
                    features[phase].append(i3d.extract_features(ip).squeeze(0).permute(1,2,3,0).data.cpu().numpy())      
                print(os.path.join(self.save_dir, name+f"__{phase}"))
                np.save(os.path.join(self.save_dir, name+f"__{phase}"), np.concatenate(features[phase], axis=0).reshape(-1, 1024))
                saved_list.append(f"{os.path.join(self.save_dir, name+'__'+phase)}.npy")
            #shutil.rmtree(dest=self.root, ignore_errors=True)
        return saved_list, elapsed_frames

if __name__ == "__main__":
    root = '/home/www/data/data/saigonmusic/Dev_AI/kiendn/dataset/horror_film/images/Bloody'
    save_dir = '/home/www/data/data/saigonmusic/Dev_AI/kiendn/dataset/horror_film/features/Bloody'
    list_img_dir = glob.glob(root+'/*')
    #list_img_dir = list_img_dir[300:]
    img_feature_extractor = ImageFeatureExtractor(list_img_dir=list_img_dir, save_dir=save_dir, num_ignore_frame=0) 
    rgb_feature_files, elapsed_frames = img_feature_extractor.extract_image_features()