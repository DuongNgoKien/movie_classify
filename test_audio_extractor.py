from pipeline.audio_feature_extract import AudioFeatureExtractor

AUDIO_FEATURE_PATH = "/home/www/data/data/saigonmusic/Dev_AI/kiendn/movie_classify/features/audio_features"

if __name__ == "__main__":
    audio_list_path = ['/home/www/data/data/saigonmusic/Dev_AI/kiendn/movie_classify/audios/tess.wav']
    audio_feature_extractor = AudioFeatureExtractor(
        audio_list_path,
        feature_save_path=AUDIO_FEATURE_PATH
    )
    audio_feature_files = audio_feature_extractor.extract_audio_features()