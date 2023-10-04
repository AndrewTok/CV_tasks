import numpy as np


def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    result = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            i_centered = (i - (size - 1)/2)#/size
            j_centered = (j - (size - 1)/2)#/size
            # print(i_centered, j_centered)
            r_2 = i_centered**2 + j_centered**2
            result[i,j] = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-1*r_2/(2*sigma**2))

    result = result/result.sum()
    return result

# из семинара
def pad_kernel(kernel, target):
   th, tw = target
   kh, kw = kernel.shape[:2]
   ph, pw = th - kh, tw - kw
   padding = [((ph+1) // 2, ph // 2), ((pw+1) // 2, pw // 2)]
   kernel = np.pad(kernel, padding)
   return kernel

def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    padded = pad_kernel(h, shape)
    return np.fft.fft2(np.fft.ifftshift(padded))

def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    mask = np.abs(H) <= threshold
    H[mask] = threshold
    H_inv = np.power(H, -1)
    H_inv[mask] = 0
    return H_inv


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """

    G = np.fft.fft2(blurred_img) 
    H = fourier_transform(h, G.shape)
    F_res = G*inverse_kernel(H, threshold)
    f = np.fft.ifft2(F_res)
    return np.abs(f)

def wiener_filtering(blurred_img, h, K=0.00005):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    H = fourier_transform(h, blurred_img.shape)
    G = np.fft.fft2(blurred_img)
    F_res = np.conj(H)/(np.abs(H)**2 + K)*G
    return np.abs(np.fft.ifft2(F_res))


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    return 20*np.log10(255/np.sqrt(np.mean(np.power(img1 - img2, 2))))


