import os
import glob
import ntpath
import sys

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def convert_mp4_to_avi(file_name, output_directory):
    input_name = file_name
    output_name = ntpath.basename(file_name)
    output = os.path.join(output_directory, output_name.replace('.mp4', '.wav', 1))
    cmd = "ffmpeg -i {} -ab 160K -ac 1 -ar 16000 -vn {}".format(input_name, output) 
    return os.popen(cmd)

def main():
    input_directory = '/home/www/data/data/saigonmusic/Dev_AI/kiendn/pytorch-i3d/data/videos'
    output_directory = '/home/www/data/data/saigonmusic/Dev_AI/kiendn/pytorch-i3d/data/audios'
    createDirectory(output_directory)
    files = glob.glob(input_directory + '/*.mp4')
    for file_name in files:
        convert_mp4_to_avi(file_name, output_directory)
        print('convert mp4 to wav :', file_name)

if __name__ == "__main__":
   main()
