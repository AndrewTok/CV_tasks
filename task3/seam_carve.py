import numpy as np





def compute_energy(img: np.ndarray):

    result = img.astype('float64')

    R = result[..., 0]
    G = result[..., 1]
    B = result[..., 2]

    YUV_img = 0.299*R + 0.587*G + 0.114*B

    extended_YUV = np.pad(YUV_img, ((1,1), (1,1)), mode = 'edge')

    deriv_X = 1/2*(np.roll(extended_YUV, (-1,0), axis = (0,1)) - np.roll(extended_YUV, (1,0), axis = (0,1)))[1 : -1, 1 : -1]

    deriv_Y = 1/2*(np.roll(extended_YUV, (0,-1), axis = (0,1)) - np.roll(extended_YUV, (0,1), axis = (0,1)))[1 : -1, 1 : -1]

    deriv_X[0, :] *= 2
    deriv_X[-1, :] *= 2

    deriv_Y[:, 0] *= 2
    deriv_Y[:, -1] *= 2

    energy = np.sqrt(np.power(deriv_X,2) + np.power(deriv_Y, 2))

    return energy



def compute_seam_matrix(energy: np.ndarray, mode: str, mask = None):
    
    additional_energy = energy.shape[0]*energy.shape[1]*256
    if mask is not None:
        mask = mask.astype('float64') * additional_energy
        energy = energy + mask.astype('float64')

    if mode == 'vertical':
        energy = energy.T

    seam_matrix = np.zeros_like(energy)
    seam_matrix[0,:] = energy[0,:]




    for i in range(1, energy.shape[0]):
        for j in range(0, energy.shape[1]):
            if j == 0:
                slice = seam_matrix[i-1, j : j+2]
            
            elif j == seam_matrix.shape[1] - 1:
                slice = seam_matrix[i-1, j-1 : j + 1]
            
            else:
                slice = seam_matrix[i-1, j-1 : j + 2]
            

            min_j = np.argmin(slice)
            min = slice[min_j]

            seam_matrix[i, j] = min + energy[i,j]
    
    if mode == 'vertical':
        seam_matrix = seam_matrix.T

    return seam_matrix


def __find_seam_mask(seam_matrix: np.ndarray, mode: str):

    if mode == 'vertical':
        seam_matrix = seam_matrix.T
    
    mask = np.zeros_like(seam_matrix, dtype='bool')


    j = np.argmin(seam_matrix[seam_matrix.shape[0] - 1]) # start
    mask[seam_matrix.shape[0] - 1, j] = True

    for i in range(seam_matrix.shape[0]-2, -1, -1):
        if j == 0:
            slice = seam_matrix[i, j : j + 2]
            delta = 0
        
        elif j == seam_matrix.shape[1] - 1:
            slice = seam_matrix[i, j - 1 : j + 1]
            delta = -1
        
        else:
            slice = seam_matrix[i, j - 1 : j + 2]
            delta = -1
        
        min_j = np.argmin(slice)
        min_j = min_j + delta + j

        mask[i, min_j] = True
        j = min_j

    if mode == 'vertical':
        mask = mask.T
    return mask

modes = {
    'vertical shrink': 'vertical',
    'horizontal shrink': 'horizontal'
}

def remove_minimal_seam(image: np.ndarray, seam_matrix: np.ndarray, mode: str, mask = None):
    mode = modes[mode]
    seam_mask = __find_seam_mask(seam_matrix, mode) 

    if mode == 'vertical':
        seam_mask = seam_mask.T
        image = image.transpose(1, 0, 2)
        if mask is not None:
            mask = mask.T
    
    seam_mask_3 = np.stack([seam_mask,seam_mask,seam_mask], axis=2)

    result = image[~seam_mask_3].reshape(seam_mask.shape[0], -1, 3) 
    if mask is not None:
        mask = mask[~seam_mask].reshape(seam_mask.shape[0], -1)

    if mode == 'vertical':
        result = result.transpose(1, 0, 2)
        seam_mask = seam_mask.T
        if mask is not None:
            mask = mask.T

    return (result.astype('uint8'), mask, seam_mask.astype('uint8'))

def seam_carve(img: np.ndarray, mode: str, mask: np.ndarray = None):
    energy = compute_energy(img)
    seam_matrix = compute_seam_matrix(energy, modes[mode], mask)
    return remove_minimal_seam(img, seam_matrix, mode, mask) 