import numpy as np


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
    

    repeating_num = (n_cols + 1) // 2
    
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

    pair_lines = __get_pair_templated_lines(n_rows, start_pixel, is_second_line_empty, is_first_line_empty).reshape(-1)
    mask = np.tile(pair_lines, repeating_num).reshape(-1, n_cols)[:n_rows,:]
    return mask


        
            




def get_bayer_mask(n_rows: int, n_cols: int):
    
    # mask = np.zeros((n_rows, n_cols, 3), dtype=bool)
    
    # total = np.ndarray

    
    red_mask = __make_one_channel_mask(0, n_rows, n_cols)
    green_mask = __make_one_channel_mask(1, n_rows, n_cols)
    blue_mask = __make_one_channel_mask(2, n_rows, n_cols)

    # print(red_mask)

    mask = np.stack([red_mask, green_mask,blue_mask], axis = 0).reshape((n_rows,n_cols,3))

    return mask

def __apply_mask(raw_img:np.ndarray, mask:np.ndarray):
    res = raw_img.copy()
    res[~mask] = 0
    return res


def get_colored_img(raw_img: np.ndarray):

    n_rows = raw_img.shape[0]
    n_cols = raw_img.shape[1]
    mask = get_bayer_mask(n_rows, n_cols).reshape(3, n_rows, n_cols)

    red_channel = __apply_mask(raw_img, mask[0])#[None,...]
    # print(red_channel)
    green_channel = __apply_mask(raw_img, mask[1])#[None,...]
    # print(green_channel)
    blue_channel = __apply_mask(raw_img, mask[2])#[None,...]
    # print(blue_channel)
    return np.stack([red_channel, green_channel, blue_channel], axis=0) #.reshape(n_rows, n_cols, -1)


def __bilinear_channel_interpolation(channel_values: np.ndarray, mask: np.ndarray):

    empty_idx = np.argwhere(~mask)
    channel_values[~mask] = None
    slice = None
    for (i,j), value in np.ndenumerate(empty_idx):
        next_slice = channel_values[i-1:i+1,j-1,j+1] 
        if slice is None:
            slice = next_slice
        else:
            slice = np.stack([slice,next_slice])
    # хз

    for idx in empty_idx:
        slice = 


    pass

def bilinear_interpolation(colored_img: np.ndarray):




    return None



