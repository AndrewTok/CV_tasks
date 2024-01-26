import numpy as np
from skimage import img_as_ubyte

def __get_mask_line_with_start_pixel(start_pixel: bool):
    seq = np.zeros(2, dtype=bool)
    seq[0] = start_pixel
    seq[1] = not start_pixel
    return seq

# channels: rgb -> 012
def __get_mask_line_template_sequence(channel: int):
    return __get_mask_line_with_start_pixel(channel != 0)
    # if channel == 0:
    #     return __get_mask_line_with_start_pixel(False)
    # return __get_mask_line_with_start_pixel(True)

def __get_one_templated_line(n_rows: int, start_pixel: bool):
    repeat_number = (n_rows + 1)//2
    line_template = __get_mask_line_with_start_pixel(start_pixel)
    return np.tile(line_template, repeat_number)[:n_rows]

def __get_pair_templated_lines(n_rows: int, start_pixel: bool, is_second_line_empty: bool, is_first_line_empty: bool):
    if is_first_line_empty:
        first_line = np.zeros(n_rows, dtype=bool)
        second_line = __get_one_templated_line(n_rows, start_pixel)
    elif is_second_line_empty:
        first_line = __get_one_templated_line(n_rows, start_pixel)
        second_line = np.zeros_like(first_line, dtype=bool)
    else:
        first_line = __get_one_templated_line(n_rows, start_pixel)
        second_line = __get_one_templated_line(n_rows, not start_pixel)

    return np.stack([first_line, second_line], axis=0)

def __make_one_channel_mask(channel: int, n_rows: int, n_cols: int):

    # mask = np.zeros((n_rows, n_cols), dtype=bool)
    # line_template = __get_mask_line_template_sequence(channel)
    # ranges = { 0: np.arange(0, n_cols-1), 1: np.arange(n_cols), 2: np.arange(1, n_cols) } # to global space?
    

    repeating_num = (n_rows + 1) // 2
    
    if channel == 0:
        start_pixel = False
        is_second_line_empty = True
        is_first_line_empty = False
    elif channel == 1:
        start_pixel = True
        is_second_line_empty = False
        is_first_line_empty = False
    else:
        start_pixel = True
        is_second_line_empty = False
        is_first_line_empty = True

    pair_lines = __get_pair_templated_lines(n_cols, start_pixel, is_second_line_empty, is_first_line_empty).reshape(-1)
    mask = np.tile(pair_lines, repeating_num).reshape(repeating_num*2, -1)[:n_rows,:n_cols]
    return mask


        
            




def get_bayer_masks(n_rows: int, n_cols: int):

    red_mask = __make_one_channel_mask(0, n_rows, n_cols)
    green_mask = __make_one_channel_mask(1, n_rows, n_cols)
    blue_mask = __make_one_channel_mask(2, n_rows, n_cols)

    masks = np.zeros((n_rows, n_cols, 3), 'bool')
    masks[..., 0] = red_mask
    masks[..., 1] = green_mask
    masks[..., 2] = blue_mask

    return masks #np.dstack([red_mask, green_mask, blue_mask])

def __apply_mask(raw_img:np.ndarray, mask:np.ndarray):
    res = raw_img.copy()
    res[~mask] = 0
    return res


def get_colored_img(raw_img: np.ndarray):

    n_rows = raw_img.shape[0]
    n_cols = raw_img.shape[1]
    mask = get_bayer_masks(n_rows, n_cols)#.reshape(3, n_rows, n_cols)

    red_channel = __apply_mask(raw_img, mask[..., 0]) #[None,...]
    # print(red_channel)
    green_channel = __apply_mask(raw_img, mask[..., 1])#[None,...]
    # print(green_channel)
    blue_channel = __apply_mask(raw_img, mask[..., 2])#[None,...]
    # print(blue_channel)
    return np.dstack([red_channel, green_channel, blue_channel]) #.reshape(n_rows, n_cols, -1)


def __bilinear_channel_interpolation(channel_values: np.ndarray, mask: np.ndarray):
    float_channel_values = channel_values.astype(dtype = 'float')
    float_channel_values[~mask] = np.NaN
    result = np.copy(channel_values)
    n_rows = channel_values.shape[0]
    n_cols = channel_values.shape[1]
    for i in range(n_rows):
        for j in range(n_cols):
            if not mask[i,j]:
                img_slice = float_channel_values[i-1 : i+2, j-1 : j+2]
                mean = np.nanmean(img_slice)
                result[i,j] = 0 if np.isnan(mean) else int(mean)
                # if img_slice.any():

    return result

    # empty_idx = np.argwhere(~mask)
    # channel_values[~mask] = None
    # slice = None
    # for (i,j), value in np.ndenumerate(empty_idx):
    #     next_slice = channel_values[i-1:i+1,j-1,j+1] 
    #     if slice is None:
    #         slice = next_slice
    #     else:
    #         slice = np.stack([slice,next_slice])
    # # хз

    # for idx in empty_idx:
    #     slice = 


    pass


def set_interpolated_pixel_value(result: np.ndarray, masks: np.ndarray, slices: np.ndarray, i: int, j: int, channel: int):
    # img_slice = float_channel_values[i-1 : i+2, j-1 : j+2]
    mean = np.nanmean(slices[..., channel])
    if not masks[i,j,channel]:
        result[i,j, channel] = 0 if np.isnan(mean) else int(mean)


def bilinear_interpolation_faster(colored_img: np.ndarray):
    n_rows = colored_img.shape[0]
    n_cols = colored_img.shape[1]

    float_channel_values = colored_img.astype(dtype = 'float')
    result = np.copy(colored_img)

    mask = get_bayer_masks(n_rows, n_cols)#.reshape(3, n_rows, n_cols)
    float_channel_values[~mask] = np.NaN

    for i in range(n_rows):
        for j in range(n_cols):
            img_slice = float_channel_values[i-1 : i+2, j-1 : j+2]
            set_interpolated_pixel_value(result, mask, img_slice, i, j, 0)
            set_interpolated_pixel_value(result, mask, img_slice, i, j, 1)
            set_interpolated_pixel_value(result, mask, img_slice, i, j, 2)
    return result

def bilinear_interpolation(colored_img: np.ndarray):

    return bilinear_interpolation_faster(colored_img)
    # n_rows = colored_img.shape[0]
    # n_cols = colored_img.shape[1]

    # mask = get_bayer_masks(n_rows, n_cols)#.reshape(3, n_rows, n_cols)




    # red_interpolated = __bilinear_channel_interpolation(colored_img[..., 0], mask[..., 0])
    # green_interpolated = __bilinear_channel_interpolation(colored_img[..., 1], mask[..., 1])
    # blue_interpolated = __bilinear_channel_interpolation(colored_img[..., 2], mask[..., 2])
# return np.dstack([red_interpolated, green_interpolated, blue_interpolated])#.astype('uint8')

G_at_R_coefs = np.array([[0,0,-1,0,0],
                        [0,0,0,0,0],
                        [-1,0,4,0,-1],
                        [0,0,0,0,0],
                        [0,0,-1,0,0]])

G_at_B_coefs = G_at_R_coefs

R_at_G_in_RB_coefs = np.array([[0,0,1/2,0,0],
                        [0,-1,0,-1,0],
                        [-1,0,5,0,-1],
                        [0,-1,0,-1,0],
                        [0,0,1/2,0,0]])

R_at_G_in_BR_coefs = np.array([[0,0,-1,0,0],
                        [0,-1,0,-1,0],
                        [1/2,0,5,0,1/2],
                        [0,-1,0,-1,0],
                        [0,0,-1,0,0]])

R_at_B_coefs = np.array([[0,0,-3/2,0,0],
                        [0,0,0,0,0],
                        [-3/2,0,6,0,-3/2],
                        [0,0,0,0,0],
                        [0,0,-3/2,0,0]])

B_at_G_in_BR_coefs = R_at_G_in_RB_coefs

B_at_G_in_RB_coefs = R_at_G_in_BR_coefs

B_at_R_coefs = R_at_B_coefs

def choose_coefs(i, j, masks, channel):
    #return coefs matrix, total weight, channel to apply coefs
    if channel == 1:
        if masks[i,j,0]:
            return G_at_R_coefs, 4, 0
        return G_at_B_coefs, 4, 2
    if channel == 0:
        if masks[i,j+1,channel] or masks[i,j-1,channel]:
            return R_at_G_in_RB_coefs, 6, 1
        if masks[i+1,j,channel] or masks[i-1,j,channel]:
            return R_at_G_in_BR_coefs, 6, 1 
        else:
            return R_at_B_coefs, 6, 2
        
    if channel == 2:
        if masks[i,j+1,channel] or masks[i,j-1,channel]:
            return B_at_G_in_BR_coefs, 6, 1
        if masks[i+1,j,channel] or masks[i-1,j,channel]:
            return B_at_G_in_RB_coefs, 6, 1
        else:
            return B_at_R_coefs, 6, 0
    return None




def set_improved_interpolated_pixel_value(result: np.ndarray, masks: np.ndarray, slices: np.ndarray, i: int, j: int, channel: int):
    # img_slice = float_channel_values[i-1 : i+2, j-1 : j+2]
    small_slices = slices[1:4,1:4,:]
    mean = np.nanmean(small_slices[..., channel])

    if not masks[i,j,channel] and slices.any():
        bilinear_val = 0 if np.isnan(mean) else mean
        coefs, weight, coefs_channel = choose_coefs(i,j,masks, channel)
        # print(coefs.shape, slices[channel].shape)
        improved_value = np.nansum(slices[..., coefs_channel]*coefs[:slices.shape[0], :slices.shape[1]])
        total_v = (np.clip(int(1/(8)*(8*bilinear_val + improved_value)), 0, 255))
        result[i,j,channel] = total_v




def improved_interpolation(raw_img: np.ndarray):

    colored_img = get_colored_img(raw_img)

    n_rows = colored_img.shape[0]
    n_cols = colored_img.shape[1]

    float_channel_values = colored_img.astype(dtype = 'float')
    result = np.copy(colored_img)

    mask = get_bayer_masks(n_rows, n_cols) #.reshape(3, n_rows, n_cols)
    float_channel_values[~mask] = np.NaN

    for i in range(1, n_rows-1):
        for j in range(1, n_cols-1):
            img_slice = float_channel_values[i-2 : i+3, j-2 : j+3]
            set_improved_interpolated_pixel_value(result, mask, img_slice, i, j, 0)
            set_improved_interpolated_pixel_value(result, mask, img_slice, i, j, 1)
            set_improved_interpolated_pixel_value(result, mask, img_slice, i, j, 2)

    return result
 

def MSE(img_pred:np.ndarray, img_gt:np.ndarray):
    return np.mean(np.power(img_pred - img_gt,2))

def compute_psnr(img_pred: np.ndarray, img_gt: np.ndarray):

    float_img_pred = img_pred.astype('float64')
    float_img_gt = img_gt.astype('float64')

    mse = MSE(float_img_pred, float_img_gt)
    if mse == 0:
        raise ValueError("Images are the same")

    return 10*np.log10(np.max(np.power(float_img_gt,2))/mse)


    pass


# masks = get_bayer_masks(2, 2)
# gt_masks = np.zeros((2, 2, 3), 'bool')
# gt_masks[..., 0] = np.array([[0, 1], [0, 0]])
# gt_masks[..., 1] = np.array([[1, 0], [0, 1]])
# gt_masks[..., 2] = np.array([[0, 0], [1, 0]])
# np.assert_ndarray_equal(actual=masks, correct=gt_masks)

# print(gt_masks.reshape((3,2,2))[0])

# print(get_bayer_masks(3,3)[...,0])

# test_channel = np.array([[0, 5, 0],[0, 0, 0], [1, 0, 1]])
# test_mask = np.array([[False, True, False],[False, False, False],[True, False, True]])
# print(__bilinear_channel_interpolation(test_channel, test_mask))

# raw_img = np.array([[8, 5, 3, 7, 1, 3],
#                      [5, 2, 6, 8, 8, 1],
#                      [9, 9, 8, 1, 6, 4],
#                      [9, 4, 2, 3, 6, 8],
#                      [5, 4, 3, 2, 8, 7],
#                      [7, 3, 3, 6, 9, 3]], dtype='uint8')

# gt_img = np.zeros((6, 6, 3), 'uint8')
# r = slice(2, -2), slice(2, -2)
# gt_img[r + (0,)] = np.array([[6, 1],
#                           [1, 0]])
# gt_img[r + (1,)] = np.array([[8, 4],
#                           [2, 3]])
# gt_img[r + (2,)] = np.array([[7, 2],
#                           [2, 2]])
# img = img_as_ubyte(improved_interpolation(raw_img))
# # assert_ndarray_equal(actual=img[r],
#                         #  correct=gt_img[r], atol=1)
# print(gt_img[r + (0,)])
# print(img[r + (0,)])
