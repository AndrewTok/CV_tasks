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
