from pipeline.audio_feature_extract import AudioFeatureExtractor
from pipeline.image_feature_extract import ImageFeatureExtractor
from pipeline.detect_scene import infer
from pytorchi3d.mp4_to_jpg import convert_mp4_to_jpg
from torchvggish.torchvggish.mp4_to_wav import convert_mp4_to_avi
import numpy as np
import opennsfw2 as n2

AUDIO_PATH = "/home/www/data/data/saigonmusic/Dev_AI/kiendn/movie_classify/audios"
IMAGE_PATH = "/home/www/data/data/saigonmusic/Dev_AI/kiendn/movie_classify/images"
IMG_FEATURE_PATH = "/home/www/data/data/saigonmusic/Dev_AI/kiendn/movie_classify/features/img_features"
AUDIO_FEATURE_PATH = "/home/www/data/data/saigonmusic/Dev_AI/kiendn/movie_classify/features/audio_features"
VIOLENCE_CHECKPOINT= "/home/www/data/data/saigonmusic/Dev_AI/kiendn/checkpoint/ckpt/violence.pkl"
HORROR_CHECKPOINT = "/home/www/data/data/saigonmusic/Dev_AI/kiendn/checkpoint/ckpt/horror.pkl"

def detect_violence(list_img_dir, audio_list_path, fps):
    #Image Feature Extraction
    img_feature_extractor = ImageFeatureExtractor(list_img_dir=list_img_dir) 
    rgb_feature_files, elapsed_frames = img_feature_extractor.extract_image_features()
    #Audio Feature Extraction
    audio_feature_extractor = AudioFeatureExtractor(audio_list_path, feature_save_path=AUDIO_FEATURE_PATH)
    audio_feature_files = audio_feature_extractor.extract_audio_features()
    
    pred = infer(HORROR_CHECKPOINT, rgb_feature_files, audio_feature_files)
    elapsed_seconds = np.array(elapsed_frames)/fps
    return pred, elapsed_seconds

def detect_horror(list_img_dir, audio_list_path, fps):
    #Image Feature Extraction
    img_feature_extractor = ImageFeatureExtractor(list_img_dir=list_img_dir) 
    rgb_feature_files, elapsed_frames = img_feature_extractor.extract_image_features()
    #Audio Feature Extraction
    audio_feature_extractor = AudioFeatureExtractor(audio_list_path, feature_save_path=AUDIO_FEATURE_PATH)
    audio_feature_files = audio_feature_extractor.extract_audio_features()
    
    pred = infer(VIOLENCE_CHECKPOINT, rgb_feature_files, audio_feature_files)
    elapsed_seconds = np.array(elapsed_frames)/fps
    return pred, elapsed_seconds
    
# def detect_pornography(video_path):
#     elapsed_seconds, nsfw_probabilities = n2.predict_video_frames(video_path)
#     elapsed_seconds = np.array(elapsed_seconds)
#     nsfw_probabilities = np.array(nsfw_probabilities)
#     return nsfw_probabilities, elapsed_seconds

def post_predictions(pred, elapsed_seconds, threshold=0.7):
    sum_prob, count, start, end = 0, 0, 0, 0
    if elapsed_seconds.ndim == 2:
        start_seconds = elapsed_seconds[:,0]
        end_seconds = elapsed_seconds[:,1]
    else:
        start_seconds = elapsed_seconds
        end_seconds = elapsed_seconds
    for i in range(pred.shape[0]):
        if pred[i] >= 0.5:
            if count == 0:
                start = start_seconds[i]
            count += 1
            sum_prob += pred[i]
        else:
            if count !=0:
                end = end_seconds[i-1]
                avg_prob = sum_prob/count
                if avg_prob >= threshold:
                    print(str(start) + " -> " + str(end))
                        
                count = 0
                sum_prob = 0
    if count != 0:
        end = end_seconds[i]
        avg_prob = sum_prob/count
        if avg_prob >= threshold:
            print(str(start) + " -> " + str(end))

if __name__ == "__main__":
    video_path = '/home/www/data/data/saigonmusic/Dev_AI/kiendn/movie_classify/videos/20240226_144614_john-rambo--doan-phim-hay-nhat.mp4'
    fps, img_dir = convert_mp4_to_jpg(video_path, IMAGE_PATH)
    list_img_dir = [img_dir]
    audio_path = convert_mp4_to_avi(video_path, AUDIO_PATH)
    audio_list_path = [audio_path]
    # fps = 24.0
    # list_img_dir = ['/home/www/data/data/saigonmusic/Dev_AI/kiendn/movie_classify/images/The Final Destination (2009) - The Escalator Kill Scene ｜ Movieclips [8pZDrxPUoyg]']
    # audio_list_path = ['/home/www/data/data/saigonmusic/Dev_AI/kiendn/movie_classify/audios/The Final Destination (2009) - The Escalator Kill Scene ｜ Movieclips [8pZDrxPUoyg]..wav']
            
    pred, elapsed_seconds = detect_violence(list_img_dir, audio_list_path, fps)
    post_predictions(pred, elapsed_seconds, threshold=0.7)