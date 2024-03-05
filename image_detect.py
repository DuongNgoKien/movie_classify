import numpy as np
import cv2
from skimage.transform import resize
import torch
import opennsfw2 as n2
import numpy as np


from pipeline.audio_feature_extract import AudioFeatureExtractor
from pipeline.image_feature_extract import ImageFeatureExtractor
from pipeline import detect_scene


VIDEO_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/videos"
AUDIO_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/audios"
IMAGE_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/images"
IMG_FEATURE_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/features/img_features"
AUDIO_FEATURE_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/features/audio_features"
SUB_PATH = "/home/www/data/data/saigonmusic/Dev_AI/manhvd/movie_classify/subs"
VIOLECE_CHECKPOINT= "/home/www/data/data/saigonmusic/Dev_AI/kiendn/checkpoint/ckpt/violence.pkl"
HORROR_CHECKPOINT = "/home/www/data/data/saigonmusic/Dev_AI/kiendn/checkpoint/ckpt/horror.pkl"
ROOT_API = "http://183.81.35.24:32774"
COMMAND_UPDATE_STATUS_API = f"{ROOT_API}/content_command/update_status"
CONTENT_UPDATE_STATUS_API = f"{ROOT_API}/content/update_status"


def detect_violence(list_img_dir, audio_list_path, fps):
    """Detect Violence from video.

    Args:
        list_img_dir (list): list of path to save all of image extracted from 
        video.
        audio_list_path (list): list of path to save all of audio extracted from
        video.
        fps (int): fps of video.

    Returns:
        Tuple[nd.array, list]: model predictions and elapsed_seconds.
    """
    #Image Feature Extraction
    img_feature_extractor = ImageFeatureExtractor(list_img_dir=list_img_dir) 
    rgb_feature_files, elapsed_frames = img_feature_extractor.extract_image_features()
    #Audio Feature Extraction
    audio_feature_extractor = AudioFeatureExtractor(
        audio_list_path,
        feature_save_path=AUDIO_FEATURE_PATH
    )
    audio_feature_files = audio_feature_extractor.extract_audio_features()
    pred = detect_scene.infer(
        VIOLECE_CHECKPOINT,
        rgb_feature_files,
        audio_feature_files
    )
    elapsed_seconds = np.array(elapsed_frames) / fps
    
    return pred, elapsed_seconds


def detect_horror(list_img_dir, audio_list_path, fps):
    """Detect Violence from video.

    Args:
        list_img_dir (list): list of path to save all of image extracted from 
        video.
        audio_list_path (list): list of path to save all of audio extracted from
        video.
        fps (int): fps of video.

    Returns:
        Tuple[nd.array, list]: model predictions and elapsed_seconds.
    """
    img_feature_extractor = ImageFeatureExtractor(list_img_dir=list_img_dir) 
    rgb_feature_files, elapsed_frames = img_feature_extractor.extract_image_features()
    #Audio Feature Extraction
    audio_feature_extractor = AudioFeatureExtractor(
        audio_list_path,
        feature_save_path=AUDIO_FEATURE_PATH
    )
    audio_feature_files = audio_feature_extractor.extract_audio_features()
    pred = detect_scene.infer(HORROR_CHECKPOINT, rgb_feature_files, audio_feature_files)
    elapsed_seconds = np.array(elapsed_frames) / fps
    
    return pred, elapsed_seconds

 
def detect_pornography(video_path):
    """Detect Violence from video.

    Args:
        video_path (str): path of video.

    Returns:
        Tuple[nd.array, list]: model predictions and elapsed_seconds.
    """
    elapsed_seconds, nsfw_probabilities = n2.predict_video_frames(video_path)
    elapsed_seconds = np.array(elapsed_seconds)
    nsfw_probabilities = np.array(nsfw_probabilities)
    
    return nsfw_probabilities, elapsed_seconds


def capture(filename, timesep, rgb, h, w, frame_interval):
    """Capture frame from video.

    Args:
        filename (str): path of video.
        timesep (int): timesep.
        rgb (): RGB.
        h (int): Height of the image.
        w (int): Width of the image.
        frame_interval (int): frame interval in for loop.

    Returns:
        Tuple(nd.array, list): frame features and list of timestamps.
    """
    cap = cv2.VideoCapture(filename)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    times = []

    for i in range(0, frame_count, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
    
        if not ret:
            break
        frm = resize(frame, (h, w, rgb))
        frm = np.expand_dims(frm, axis=0)
        frm = np.moveaxis(frm, -1, 1)
        if(np.max(frm) > 1):
            frm = frm / 255.0
        frames.append(frm)
        timestamp = i / fps
        times.append(timestamp)
        
    cap.release()
    
    num_frames = (np.shape(frames)[0] // timesep) * timesep
    
    frames = np.vstack(frames[:num_frames])
    torch_frames = torch.from_numpy(
        np.float32(frames)
    ).view(-1, timesep, rgb, h, w)
    times = times[:num_frames]
    l_intervals = []
    start = times[0]
    for i in range(1, num_frames):
        if i % timesep == 0:
            end = times[i - 1]
            l_intervals.append((start, end))
            start = times[i]
    l_intervals.append((start, times[num_frames - 1]))
    
    return torch_frames, l_intervals


def resume_checkpoint(resume_path, model):
    """Load model checkpoint.

    Args:
        resume_path (str or PathLike): Path of model checkpoint.
        model (torch.nn.Module): Model.
    """
    print(f'Loading checkpoint : {resume_path}')
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model'])