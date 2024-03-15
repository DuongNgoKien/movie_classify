import numpy as np
import torch

from decord import VideoReader, cpu
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification


def sample_frame_indices(clip_len, frame_sample_rate, video_path, device):
    feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0)) 
    seg_len = len(vr)
    fps = vr.get_avg_fps()

    converted_len = int(clip_len * frame_sample_rate)
    n_samples = seg_len // converted_len
    end_idx = n_samples * converted_len - frame_sample_rate
    str_idx = 0
    index = np.linspace(str_idx, end_idx, num=n_samples*clip_len)
    index = index.astype(np.int64)
    re_index = index.reshape(-1,clip_len)
    pixel_values, sample = [], []
    
    for i in range(re_index.shape[0]):
        buffer = vr.get_batch(re_index[i]).asnumpy()
        for j in range(buffer.shape[0]):
            sample.append(buffer[j])
            if j == clip_len-1:
                encoding = feature_extractor([sample], return_tensors="pt")
                pixel_values.append(encoding.pixel_values.to(device)[0])
                sample = []
        
    index = index.reshape((-1, clip_len))
    vr.seek(0)  
    return pixel_values, index, fps
  

# create a list of NumPy arrays
def infer(video_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pixel_values, index, fps = sample_frame_indices(clip_len=16, frame_sample_rate=4, video_path=video_path, device=device)
    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-large-finetuned-kinetics")
    model.to(device)
    
    # forward pass
    probabilities = []
    with torch.no_grad():
        for i in range(len(pixel_values)):
            outputs = model(pixel_values[i].unsqueeze(0))
            logits = torch.nn.functional.softmax(outputs.logits, dim=1)
            smoke_drink_scores = (logits[:,100] + logits[:,101] + logits[:,102] + logits[:,316] + logits[:,317])
            probabilities.append(smoke_drink_scores.item())
      
    del model
    
    elapsed_seconds = []
    for i in range(index.shape[0]):
        start = index[i][0] / fps
        end = index[i][-1] / fps
        elapsed_seconds.append([start, end])
    
    return np.array(probabilities), np.array(elapsed_seconds)