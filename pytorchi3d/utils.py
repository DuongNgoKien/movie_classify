import os


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            return False
        else:
            return True
    except OSError:
        print("Error: Failed to create the directory.")