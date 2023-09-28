import numpy as np
# from scipy.ndimage import shift



# r_start_row = 0
# g_start_row = 0

def shift_img(img: np.ndarray, u: int, v: int):
    return np.roll(img, (u,v), axis = (0,1))

def get_new_coord(size: int, prev: int,  shift: int):
    return (prev + shift) % size

def get_new_point(size: (int,int), prev: (int,int), shift_uv: (int,int), i_axes_shift: int):

    new_i = get_new_coord(size[0], prev[0] + i_axes_shift, shift_uv[0])
    new_j = get_new_coord(size[1], prev[1], shift_uv[1])
    return [new_i, new_j]

def split_img_into_channels(_img: np.ndarray):
    delta = 3 - _img.shape[0] % 3 
    img = np.pad(_img, ((0, delta),(0,0))) #_img.copy() #
    n_rows = img.shape[0] // 3

    splited = img[:n_rows*3,:].reshape(3, -1, img.shape[1]).transpose(1, 2, 0)
    tmp = splited[..., 2].copy()
    splited[..., 2] = splited[..., 0].copy()
    splited[..., 0] = tmp

    return splited

def cut_splited_imgs(splited_imgs: np.ndarray):
    n_rows = splited_imgs.shape[0]
    n_cols = splited_imgs.shape[1]

    delta_rows = int(n_rows*0.1)
    delta_cols = int(n_cols*0.1)

    # r_start_row = delta_rows + 2*n_rows
    # g_start_row = delta_rows + n_rows

    return splited_imgs[delta_rows:-delta_rows, delta_cols:-delta_cols,:]

def get_correlation_matrix(img1: np.ndarray, img2: np.ndarray):
    C = np.abs(np.fft.ifft2(np.fft.fft2(img1.astype('float64'))*np.conj(np.fft.fft2(img2.astype('float64')))))
    return C

def get_shift(correlation_matrix: np.ndarray):
    shift = np.where(correlation_matrix == correlation_matrix.max()) #np.argmax(C_red_green, axis = 1)
    shift_u, shift_v = shift[0][0], shift[1][0]
    n_rows = correlation_matrix.shape[0]
    n_cols = correlation_matrix.shape[1]
    # shift_u = shift_u if shift_u < n_rows//2 else n_rows - shift_u
    # shift_v= shift_v if shift_v < n_cols//2 else n_cols - shift_v
    return [shift_u, shift_v]

def align(img: np.ndarray, g_coord: (int,int)) -> (np.ndarray, (int, int), (int,int)):
    # print(img.shape)
    splited_imgs = split_img_into_channels(img)
    splited_imgs = cut_splited_imgs(splited_imgs)
    C_red_green = get_correlation_matrix(splited_imgs[..., 1], splited_imgs[..., 0])
    C_blue_green = get_correlation_matrix(splited_imgs[..., 1], splited_imgs[..., 2])

    n_rows = splited_imgs.shape[0]
    n_cols = splited_imgs.shape[1]
    # print(C_red_green.shape)
    # red_green_shift = np.where(C_red_green == C_red_green.max()) #np.argmax(C_red_green, axis = 1)
    # red_green_shift = red_green_shift[0][0], red_green_shift[1][0]
    # blue_green_shift = np.where(C_blue_green == C_blue_green.max()) #np.argmax(C_red_green, axis = 1)
    # blue_green_shift = blue_green_shift[0][0], blue_green_shift[1][0]

    red_green_shift = get_shift(C_red_green)
    blue_green_shift = get_shift(C_blue_green)

    # print(red_green_shift)
    splited_imgs[..., 0] = np.roll(splited_imgs[..., 0], red_green_shift, axis = (0, 1))
    splited_imgs[..., 2] = np.roll(splited_imgs[..., 2], blue_green_shift, axis = (0,1))

    # (b_row, b_col) = get_new_coord(n_rows, g_coord[0], blue_green_shift[0])

    size = (n_rows, n_cols)

    height = img.shape[0]//3 + 1

    height_cropped = splited_imgs.shape[0]


    blue_coord = get_new_point((height_cropped, n_cols), g_coord, blue_green_shift, -1*height)
    # blue_coord[0] -= n_rows

    red_coord = get_new_point((height_cropped, n_cols), g_coord, red_green_shift, -1*height)# + (n_rows, 0)
    # red_coord[0] -= n_rows
    red_coord[0] += 2*height
    # print(blu)

    return splited_imgs, blue_coord, red_coord 





# # align(None, None)
# m1 = np.arange(9).reshape((3,3))
# m2 = np.random.uniform(0, 10, (3, 3))#.reshape((3,3))
# C = get_correlation_matrix(m1, np.roll(m1, (2,0), axis = (0,1)))
# print(C)
from skimage.io import imread, imsave
import matplotlib.pyplot as plt
from skimage.draw import disk

img = imread('public_tests/03_test_img_input/img.png') #join(data_dir, 'img.png'), plugin='matplotlib')

g_coord = (508, 237)
result = align(img, g_coord)

rr_g, cc_g = disk(g_coord, 5)
img[rr_g, cc_g] = 100


rr_r, cc_r = disk(result[2], 5)
img[rr_r, cc_r] = 100


rr_b, cc_b = disk(result[1], 5)
img[rr_b, cc_b] = 100

plt.imshow(img)
plt.show()

plt.imshow(result[0])
plt.show()


