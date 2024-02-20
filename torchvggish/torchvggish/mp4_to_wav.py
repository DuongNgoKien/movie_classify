import os
import glob
import ntpath

def convert_mp4_to_avi(file_name, output_directory):
    input_name = file_name
    output_name = ntpath.basename(file_name)
    extension = output_name.split('.')[-1]
    output = os.path.join(output_directory, output_name.replace(extension, 'wav', 1))
    # cmd = 'ffmpeg -i "{input}" -c:v libx264 -c:a libmp3lame -b:a 384K "{output}"'.format(
    #                                                 input = input_name, 
    #                                                 output = output)

    cmd = 'ffmpeg -i "{}" -ab 160K -ac 1 -ar 16000 -vn "{}" -y'.format(input_name, output) #os.path.join(this_path, save_file_id))

    os.system(cmd)
    return  output


def main():
    input_directory = '/opt/ml/pytorch-i3d/data'#sys.argv[1]
    output_directory = '/opt/ml/pytorch-i3d/data_wav/'
    files = glob.glob(input_directory + '/*.mp4')
    file_name = '/opt/ml/pytorch-i3d/data/testset.mp4'
    convert_mp4_to_avi(file_name, output_directory)
    # convert_mp4_to_avi(files, output_directory)
    # for file_name in files:
    #     try:
    #         print(file_name)
    #         # convert_mp4_to_avi(file_name, output_directory)
    #     except:
    #         raise

if __name__ == "__main__":
   main()
