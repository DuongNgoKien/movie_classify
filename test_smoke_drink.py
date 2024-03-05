from pipeline import smoke_drink_detect, politic_detect

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
            
pred, elapsed_time = politic_detect.infer(img_dir="/home/www/data/data/saigonmusic/Dev_AI/kiendn/movie_classify/images/Haunted House Scene ｜ IT (2017) Horror, Movie CLIP HD [Lt0Wm_s9i6s]",
                                         model_path="/home/www/data/data/saigonmusic/Dev_AI/kiendn/protest-detection-violence-estimation/model_best.pth.tar")

post_predictions(pred, elapsed_time, threshold=0.7)