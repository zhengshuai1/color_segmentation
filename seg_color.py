import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os


def save_seg_result(img_dirs):

    save_path = './result'
    os.makedirs(save_path, exist_ok=True)
    for img_ind, img_dir in enumerate(img_dirs):
        img = cv2.imread(img_dir)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_list, pixel_num_list = seg_simple_colors(img, hsv)
        print(pixel_num_list)
        rows = 3
        cols = 2
        plt.figure()
        for ind in range(len(img_list)):
            plt.subplot(rows, cols, ind+1)
            plt.imshow(img_list[ind][:, :, ::-1])
        plt.savefig(f'./result/result{img_ind+1}.png', dpi=300)
        plt.show()


def seg_simple_colors(img, hsv):
    '''

    :param img:
    :param hsv:
    :return:
        img_list：
        pixel_num_list： pixel num by order red, blue, yellow, orange, green
    '''
    # lower red
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    # upper red
    # lower_red2 = np.array([170,50,50])
    lower_red2 = np.array([155, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # define range of blue color in HSV
    lower_blue = np.array([95, 100, 100])
    upper_blue = np.array([115, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    res_blue = cv2.bitwise_and(img, img, mask=mask)
    blue_pixel_num = np.transpose(np.nonzero(mask)).shape[0]

    lower_green = np.array([50, 100, 100])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    res_green = cv2.bitwise_and(img, img, mask=mask)
    green_pixel_num = np.transpose(np.nonzero(mask)).shape[0]

    lower_orange = np.array([13, 51, 30])
    upper_orange = np.array([21, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    res_orange = cv2.bitwise_and(img, img, mask=mask)
    orange_pixel_num = np.transpose(np.nonzero(mask)).shape[0]

    lower_yellow = np.array([26, 38, 0])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    res_yellow = cv2.bitwise_and(img, img, mask=mask)
    # print( np.transpose(np.nonzero(mask)).shape,'---')
    yellow_pixel_num = np.transpose(np.nonzero(mask)).shape[0]

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(img, img, mask=mask)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    res2 = cv2.bitwise_and(img, img, mask=mask2)
    res_red = res + res2
    mask_red = mask + mask2
    # print(f'max red is {np.max(mask_red)}, {mask_red.shape}')
    red_pixel_num = np.transpose(np.nonzero(mask_red)).shape[0]

    img_list = [img, res_red, res_blue, res_yellow, res_orange, res_green]
    pixel_num_list = [red_pixel_num, blue_pixel_num, yellow_pixel_num, orange_pixel_num, green_pixel_num]
    return img_list, pixel_num_list


def compute_seg_pixel(img_dirs):
    color_order = ['red', 'blue', 'yellow', 'orange', 'green']
    for img_ind, img_dir in enumerate(img_dirs):
        img = cv2.imread(img_dir)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_list, pixel_num_list = seg_simple_colors(img, hsv)
        print(f'img shape is {img.shape}')
        for i in range(len(color_order)):
            print(f'pixel num of {color_order[i]} is {pixel_num_list[i]}')


def main():
    img_path = './images/test/'
    img_dirs = glob.glob(img_path + "t*.png")
    img_dirs.sort()
    # save_seg_result(img_dirs)
    compute_seg_pixel(img_dirs)


if __name__ == '__main__':
    main()
