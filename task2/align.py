import numpy as np
# from scipy.ndimage import shift
# from skimage.io import imread, imsave
# import matplotlib.pyplot as plt
# from skimage.draw import disk




edges_coef = 10

def backward_img_transform(splited_img: np.ndarray, delta: int, pad: int):
    splited_img = splited_img.copy()
    tmp = splited_img[..., 2].copy()
    splited_img[..., 2] = splited_img[..., 0].copy()
    splited_img[..., 0] = tmp
    
    orig = np.pad(splited_img, ((delta, delta), (delta,delta), (0,0)))

    orig = orig.transpose(2, 0, 1).reshape(orig.shape[0]*3, -1)

    return orig[:-pad,...]

def get_img_height(shape: np.ndarray):
    return (shape[0] + 2) // 3


def forward_transform_img(_img: np.ndarray):
    pad = 3 - _img.shape[0] % 3 
    img = np.pad(_img, ((0, pad),(0,0)), mode='empty')
    n_rows = img.shape[0] // 3

    splited = img[:n_rows*3,:].reshape(3, -1, img.shape[1]).transpose(1, 2, 0)
    tmp = splited[..., 2].copy()
    splited[..., 2] = splited[..., 0].copy()
    splited[..., 0] = tmp

    return splited, pad, n_rows


def to_local(channel: int, glob_i: int, glob_j: int, delta_rows: int, delta_cols: int, source_img_height: int):
    loc_i = glob_i - channel*source_img_height - delta_rows
    loc_j = glob_j - delta_cols
    return [loc_i, loc_j]

def to_global(channel: int, loc_i: int, loc_j: int, delta_rows: int, delta_cols:int, source_img_height: int):
    glob_i = loc_i + channel*source_img_height + delta_rows
    glob_j = loc_j + delta_cols
    return [glob_i, glob_j]

def get_channel_coord_from_green(channel: int, g_coord, delta_rows: int, delta_cols: int, source_img_height: int, splited_shape: np.ndarray, shift):

    g_loc = to_local(1, g_coord[0], g_coord[1], delta_rows, delta_cols, source_img_height)

    channel_loc_i = get_new_coord(splited_shape[0], g_loc[0], shift[0])
    channel_loc_j = get_new_coord(splited_shape[1], g_loc[1], shift[1])

    channel_glob = to_global(channel, channel_loc_i, channel_loc_j, delta_rows, delta_cols, source_img_height)
    return channel_glob

def shift_img(img: np.ndarray, u: int, v: int):
    return np.roll(img, (u,v), axis = (0,1))

def get_new_coord(size: int, prev: int,  shift: int):
    return (prev - shift) # % size

def get_new_point(size: (int,int), prev: (int,int), shift_uv: (int,int)):

    new_i = get_new_coord(size[0], prev[0], shift_uv[0])
    new_j = get_new_coord(size[1], prev[1], shift_uv[1])
    return [new_i, new_j]

def cut_splited_imgs(splited_imgs: np.ndarray):
    n_rows = splited_imgs.shape[0]
    n_cols = splited_imgs.shape[1]

    delta_rows = int(n_rows//edges_coef)
    delta_cols = int(n_cols//edges_coef)

    return splited_imgs[delta_rows:-delta_rows, delta_cols:-delta_cols,:], delta_rows, delta_cols

def get_correlation_matrix(img1: np.ndarray, img2: np.ndarray):
    C = np.abs(np.fft.ifft2(np.fft.fft2(img1)*np.conj(np.fft.fft2(img2))))
    return C

def get_shift(correlation_matrix: np.ndarray):
    shift = np.where(correlation_matrix == correlation_matrix.max())
    shift_u, shift_v = shift[0][0], shift[1][0]
    n_rows = correlation_matrix.shape[0]
    n_cols = correlation_matrix.shape[1]
    shift_u = shift_u if shift_u < n_rows//2 else -1*(n_rows - shift_u)
    shift_v= shift_v if shift_v < n_cols//2 else -1*(n_cols - shift_v)
    return [shift_u, shift_v]

def align(img: np.ndarray, g_coord: (int,int)) -> (np.ndarray, (int, int), (int,int)):
    transformed_imgs, pad, height = forward_transform_img(img)
    splited_imgs, delta_rows, delta_cols = cut_splited_imgs(transformed_imgs)


    С_0 = get_correlation_matrix(splited_imgs[..., 1], splited_imgs[..., 0])
    С_2 = get_correlation_matrix(splited_imgs[..., 1], splited_imgs[..., 2])

    C_0_shift = get_shift(С_0)
    C_2_shift = get_shift(С_2)

    splited_imgs[..., 0] = np.roll(splited_imgs[..., 0], C_0_shift, axis = (0, 1))
    splited_imgs[..., 2] = np.roll(splited_imgs[..., 2], C_2_shift, axis = (0, 1))

    blue_coord = get_channel_coord_from_green(0, g_coord, delta_rows, delta_cols, height, splited_imgs.shape, C_2_shift)
    red_coord = get_channel_coord_from_green(2, g_coord, delta_rows, delta_cols, height, splited_imgs.shape, C_0_shift)
    
    return splited_imgs, blue_coord, red_coord 





# # align(None, None)
# m1 = np.arange(9).reshape((3,3))
# m2 = np.random.uniform(0, 10, (3, 3))#.reshape((3,3))
# C = get_correlation_matrix(m1, np.roll(m1, (2,0), axis = (0,1)))
# print(C)


# img = imread('public_tests/00_test_img_input/img.png') #join(data_dir, 'img.png'), plugin='matplotlib')

# g_coord = (508, 237)
# result = align(img, g_coord)

# # orig = backward_img_transform(result[0], get_img_height(img.shape)//edges_coef)

# rr_g, cc_g = disk(g_coord, 3)
# img[rr_g, cc_g] = 100


# rr_r, cc_r = disk(result[2], 3)
# img[rr_r, cc_r] = 100


# rr_b, cc_b = disk(result[1], 3)
# img[rr_b, cc_b] = 100



# plt.imshow(img)
# plt.show()

# # result[0][...,2] = 0 #np.zeros((result.shape[0], res))

# plt.imshow(result[0])
# plt.show()


