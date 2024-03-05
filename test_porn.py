import opennsfw2 as n2
import numpy as np

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
            
def detect_pornography(video_path):
    elapsed_seconds, nsfw_probabilities = n2.predict_video_frames(video_path)
    elapsed_seconds = np.array(elapsed_seconds)
    nsfw_probabilities = np.array(nsfw_probabilities)
    
    return nsfw_probabilities, elapsed_seconds

if __name__ == "__main__":
    a, b = detect_pornography("/home/www/data/data/saigonmusic/Dev_AI/kiendn/movie_classify/videos/20240227_135815_-red--the-ambush--military-action-short.mp4")
    print(np.max(a))
    post_predictions(a,b,0.5)