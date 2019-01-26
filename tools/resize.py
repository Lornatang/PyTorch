import os.path
import glob
import cv2


def convert_size(file, out_dir, width, height):
    raw_img = cv2.imread(file)
    try:
        new_img = cv2.resize(raw_img, (width, height))
        cv2.imwrite(os.path.join(out_dir, os.path.basename(file)), new_img)
    except Exception as e:
        print(e)


class RESIZE(object):
    """use one key to reformat the image size.

    para:
        work_dir: A directory for work.
        high: Pictures that need to be modified high.
        width: Pictures that need to be modified width.
    return:
        None
    """

    def __init__(self, work_dir, high, width):
        self.work_dir = work_dir
        self.high = high
        self.width = width

        if self.work_dir is None:
            raise Exception('Work dir cannot be empty!')
        elif self.high is None:
            raise Exception('Image high cannot be empty!')
        elif self.width is None:
            raise Exception('Image width cannot be empty!')

        for root in os.listdir(work_dir):
            dirs = os.path.join(work_dir, root)
            print(f"'{dirs}' are being resized!")
            for img in glob.glob(dirs + '/*'):
                if img is None:
                    raise Exception('The images in the directory must be in JPG/PNG format.')
                convert_size(img, dirs, high, width)
            print(f"'{dirs}' has done!")


RESIZE('../../../data/PASCAL/P2006/train', 224, 224)