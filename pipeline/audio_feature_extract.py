import numpy as np
import glob
from pathlib import Path 

import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from torchvggish.torchvggish import vggish

class AudioFeatureExtractor():
    def __init__(self, audio_path = 'data/audios',
                       feature_save_path = 'data/audio_features'):
        self.audio_path = audio_path
        self.feature_save_path = feature_save_path

    def extract_audio_features(self):
        audio = self.audio_path
        audio_name = Path(audio).stem

        # Initialise model and download weights
        model_urls = {
            'vggish': 'https://github.com/harritaylor/torchvggish/'
            'releases/download/v0.1/vggish-10086976.pth',
            'pca': 'https://github.com/harritaylor/torchvggish/'
            'releases/download/v0.1/vggish_pca_params-970ea276.pth'
            }
        embedding_model = vggish.VGGish(urls=model_urls, postprocess=False)
        embedding_model.eval()
        
        audio_list_file = []
        
        # if os.path.exists(os.path.join(self.feature_save_path, audio_name[:-4])+".npy"):
        #     return os.path.join(self.feature_save_path, audio_name[:-4])+".npy"

        embeddings = embedding_model.forward(audio)
        np.save(os.path.join(self.feature_save_path, audio_name[:-4]),embeddings.detach().cpu().numpy())
        path_save = (os.path.join(self.feature_save_path, audio_name[:-4])+".npy")
            
        audio_list_file.append(path_save)
        
        return audio_list_file