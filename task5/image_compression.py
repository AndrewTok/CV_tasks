import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio
# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def clip(img):
    return np.clip(img, 0, 255)

def pca_compression(matrix, p):
    """ Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """
    
    # Your code here
    
    # Отцентруем каждую строчку матрицы
    means = np.mean(matrix, axis = 1)[:, None]
    matrix = matrix.copy() - means
    

    # Найдем матрицу ковариации
    cov_matrix = np.cov(matrix)
    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
    # Посчитаем количество найденных собственных векторов
    eigen_v_count = eigen_vectors.shape[1]
    # Сортируем собственные значения в порядке убывания
    sorting_order = np.argsort(-1*eigen_values)
    sorted_eigen_values = eigen_values[sorting_order]
    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    sorted_eighen_vectors = eigen_vectors[:,sorting_order]
    # Оставляем только p собственных векторов
    p_eigen_vectors = sorted_eighen_vectors[:,:p]
    # Проекция данных на новое пространство

    
    projection = np.dot(p_eigen_vectors.T, matrix)
    return p_eigen_vectors, projection, means[:,0]


def pca_decompression(compressed):
    """ Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """
    
    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!
        eig_vec, proj, means = comp
        # print(means)
        decomp_i = np.matmul(eig_vec,proj) + means[:,None]
        result_img.append(decomp_i)
    res = np.clip(np.array(result_img).transpose(1, 2, 0), 0, 255)
    # print(res[...,0])
    # print(res[...,1])
    # print(res[...,2])
    return res.astype('uint8')


def pca_visualize():
    plt.clf()
    img = imread('cat.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            compressed.append(pca_compression(img[...,j], p))
            pass
        decomp = pca_decompression(compressed)
        axes[i // 3, i % 3].imshow(clip(np.array(decomp)).astype('uint8'))
        axes[i // 3, i % 3].set_title('Компонент: {}'.format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """ Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """



    R = img[..., 0]
    G = img[..., 1]
    B = img[..., 2]

    Y = 0 + 0.299*R + 0.587*G + 0.114*B
    Cb = 128 - 0.1687*R - 0.3313*G + 0.5*B
    Cr = 128 + 0.5*R - 0.4187 * G  - 0.0813*B
    return np.stack([Y, Cb, Cr], axis=2)


def ycbcr2rgb(img):
    """ Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """
    
    # Your code here

    Y = img[..., 0]
    Cb = img[..., 1]
    Cr = img[..., 2]

    R = Y + 1.402*(Cr - 128)
    G = Y - 0.34414*(Cb - 128) - 0.71414*(Cr - 128) 
    B = Y + 1.77*(Cb - 128)
    
    return np.stack([R,G,B], axis = 2)


def get_gauss_1():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here
    ycbcr = rgb2ycbcr(rgb_img)
    # gaussian_filter
    r = 10
    Cb_blurred = gaussian_filter(ycbcr[..., 1], r)
    Cr_blurred = gaussian_filter(ycbcr[..., 2], r)
    ycbcr[..., 1] = Cb_blurred
    ycbcr[..., 2] = Cr_blurred

    rgb = ycbcr2rgb(ycbcr)
    plt.imshow(clip(rgb).astype('uint8'))
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here
    ycbcr = rgb2ycbcr(rgb_img)
    # gaussian_filter
    r = 10
    Y_blured = gaussian_filter(ycbcr[...,0], r)
    ycbcr[...,0] = Y_blured
    rgb = ycbcr2rgb(ycbcr)
    plt.imshow(clip(rgb).astype('uint8'))

    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [A // 2, B // 2, 1]
    """

    return gaussian_filter(component, 10)[0::2,0::2]

def __alpha(u: int):
    if u == 0:
        return 1/np.sqrt(2)
    return 1

def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """

    # Your code here

    i_indices = np.arange(8)
    i_indices = np.tile(i_indices[:,None], 8)
    j_indices = i_indices.T

    G = np.zeros((8,8))

    for u in range(8):
        for v in range(8):
            alpha_v = __alpha(v)
            alpha_u = __alpha(u)
            cos_1 = np.cos((2*i_indices + 1)/16*u*np.pi)
            cos_2 = np.cos((2*j_indices+ 1)/16*v*np.pi)
            sum = np.sum(block*cos_1*cos_2)
            G[u,v] = 1/4*alpha_v*alpha_u*sum
    return G


# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """
    
    # Your code here
    
    return np.round(block/quantization_matrix)


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100

    if q >= 1 and q < 50:
        S = 5000/q
    elif q >= 50 and q <= 99:
        S = 200 - 2*q
    else:
        S = 1

    Q = np.floor((50 + S*default_quantization_matrix.astype('float64'))/100)
    Q[Q <= 0] = 1

    return Q


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """
    
    num_rows = block.shape[0]
    num_cols = block.shape[1]
    cur_row,cur_col = 0,0
    cur_index = 0
    result =  [] 
    # source https://github.com/xaviraol/JPEG/blob/master/jpeg/zigzag.m
    while cur_row<=num_rows and cur_col<=num_cols:
        if cur_row==0 and (cur_row+cur_col) % 2==0 and cur_col != num_cols - 1:
            result.append(block[cur_row,cur_col]) 
            cur_col=cur_col+1							#move right at the top
            cur_index=cur_index+1
            
        elif cur_row==num_rows-1 and (cur_row+cur_col) % 2 !=0 and cur_col !=num_cols-1:
            result.append(block[cur_row,cur_col]) 
            cur_col=cur_col+1						#move right at the bottom
            cur_index=cur_index+1
            
        elif cur_col==0 and (cur_row+cur_col)%2 !=0 and cur_row!=num_rows-1:
            result.append(block[cur_row,cur_col]) 
            cur_row=cur_row+1 #move down at the left
            cur_index=cur_index+1
            
        elif cur_col==num_cols-1 and (cur_row+cur_col) %2 ==0 and cur_row!=num_rows-1:
            result.append(block[cur_row,cur_col])
            cur_row=cur_row+1					#move down at the right
            cur_index=cur_index+1
            
        elif cur_col!=0 and cur_row!=num_rows-1 and (cur_row+cur_col) %2 !=0:
            result.append(block[cur_row,cur_col])
            cur_row=cur_row+1	
            cur_col=cur_col-1	#move diagonally left down
            cur_index=cur_index+1
            
        elif cur_row!=0 and cur_col!=num_cols-1 and (cur_row+cur_col) % 2==0:
            result.append(block[cur_row,cur_col])
            cur_row=cur_row-1		
            cur_col=cur_col+1	#move diagonally right up
            cur_index=cur_index+1
            
        elif cur_row==num_rows-1 and cur_col==num_cols-1:
            result.append(block[cur_row,cur_col]) 
            break						
        
    
    return result


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """
    # Your code here
    count = 0
    new_el = True
    res = []
    for i in range(len(zigzag_list)):
        val = zigzag_list[i]
        if val != 0:
            res.append(val)
        else:
            if new_el:
                res.append(val)
                new_el = False
                count = 0
            count += 1
            if i + 1 == len(zigzag_list) or zigzag_list[i+1] != 0:
                res.append(count)
                new_el = True
    return res


def __split_into_blocks(component: np.ndarray, quantization_matrix: np.ndarray):
    i_blocks_count = component.shape[0]//8
    j_blocks_count = component.shape[1]//8
    blocks = []
    for i in range(i_blocks_count):
        for j in range(j_blocks_count):
            block = component[i*8:(i+1)*8,j*8:(j+1)*8]-128

            dct_block = dct(block)
            dct_block = quantization(dct_block, quantization_matrix)

            zig_zag = zigzag(dct_block)       

            compressed = compression(zig_zag) 

            blocks.append(compressed)
    return blocks
            

def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Your code here
    
    # Переходим из RGB в YCbCr
    YCbCr = rgb2ycbcr(img)
    # Уменьшаем цветовые компоненты
    down_Cb = downsampling(YCbCr[...,1])
    down_Cr = downsampling(YCbCr[...,2])
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    
    blocks = []
    blocks.append(__split_into_blocks(YCbCr[...,0], quantization_matrixes[0]))
    blocks.append(__split_into_blocks(down_Cb, quantization_matrixes[1]))
    blocks.append(__split_into_blocks(down_Cr, quantization_matrixes[1]))


    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие

    return blocks


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """
    
    # Your code here

    result = []
    i = 0
    while i < len(compressed_list):
        val = compressed_list[i]
        if val == 0:
            for k in range(compressed_list[i+1]):
                result.append(val)
            i+=2
        else:
            result.append(val)
            i+=1

    return result


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """

    # Your code here

    cur_row,cur_col = 7,7

    num_rows = 8
    num_cols = 8
    block = np.zeros((num_rows, num_cols))
    cur_row,cur_col = 0,0
    cur_index = 0
    while cur_row<=num_rows and cur_col<=num_cols:
        if cur_row==0 and (cur_row+cur_col) % 2==0 and cur_col != num_cols - 1:
            block[cur_row,cur_col] = input[cur_index]
            cur_col=cur_col+1							#move right at the top
            cur_index=cur_index+1
            
        elif cur_row==num_rows-1 and (cur_row+cur_col) % 2 !=0 and cur_col !=num_cols-1:
            block[cur_row,cur_col] = input[cur_index]
            cur_col=cur_col+1						#move right at the bottom
            cur_index=cur_index+1
            
        elif cur_col==0 and (cur_row+cur_col)%2 !=0 and cur_row!=num_rows-1:
            block[cur_row,cur_col] = input[cur_index]
            cur_row=cur_row+1 #move down at the left
            cur_index=cur_index+1
            
        elif cur_col==num_cols-1 and (cur_row+cur_col) %2 ==0 and cur_row!=num_rows-1:
            block[cur_row,cur_col] = input[cur_index]
            cur_row=cur_row+1					#move down at the right
            cur_index=cur_index+1
            
        elif cur_col!=0 and cur_row!=num_rows-1 and (cur_row+cur_col) %2 !=0:
            block[cur_row,cur_col] = input[cur_index]
            cur_row=cur_row+1	
            cur_col=cur_col-1	#move diagonally left down
            cur_index=cur_index+1
            
        elif cur_row!=0 and cur_col!=num_cols-1 and (cur_row+cur_col) % 2==0:
            block[cur_row,cur_col] = input[cur_index]
            cur_row=cur_row-1		
            cur_col=cur_col+1	#move diagonally right up
            cur_index=cur_index+1
            
        elif cur_row==num_rows-1 and cur_col==num_cols-1:	
            block[cur_row,cur_col] = input[cur_index]
            break						
    
    return block


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """
    

    return block*quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """

    u_indices = np.arange(8)
    u_indices = np.tile(u_indices[:,None], 8)
    v_indices = u_indices.T

    f = np.zeros((8,8))

    for i in range(8):
        for j in range(8):
            alpha_v = np.ones_like(f)  #__alpha(v_indices)
            alpha_v[v_indices == 0] = 1/np.sqrt(2)
            alpha_u = np.ones_like(f)
            alpha_u[u_indices == 0] = 1/np.sqrt(2)
            cos_1 = np.cos((2*i + 1)/16*u_indices*np.pi)
            cos_2 = np.cos((2*j+ 1)/16*v_indices*np.pi)
            sum = np.sum(alpha_v*alpha_u*block*cos_1*cos_2)
            f[i,j] = 1/4*sum

    return np.round(f)


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """
    
    # Your code here
    result = np.zeros((2*component.shape[0], 2*component.shape[1]))
    result[::2,::2] = component

    shifted_i = np.roll(result, (1, 0), axis=(0,1))
    added_i = result + shifted_i
    shifted_j = np.roll(added_i, (0,1), axis=(0,1))
    result = added_i + shifted_j

    return result

def __make_component(result_shape:tuple, blocks:np.ndarray):
    result = np.zeros(result_shape)
    blocks_count_i = result_shape[0]//8
    blocks_count_j = result_shape[1]//8
    for block_idx in range(len(blocks)):
        block_i = block_idx // blocks_count_j
        block_j = block_idx - block_i*blocks_count_j
        result[block_i*8:(block_i+1)*8,block_j*8:(block_j+1)*8] = blocks[block_idx]
    return result

def __unpack_blocks(compressed_blocks: list, quantization_matrix: np.ndarray):
    unpucked = []
    for block_idx in range(len(compressed_blocks)):
        block = compressed_blocks[block_idx]
        block = inverse_compression(block)
        block = inverse_zigzag(block)
        block = inverse_quantization(block, quantization_matrix)
        block = inverse_dct(block)
        block += 128
        unpucked.append(block)
    return unpucked

def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """

    # Your code here
    Y_blocks = __unpack_blocks(result[0], quantization_matrixes[0])
    Cb_blocks = __unpack_blocks(result[1], quantization_matrixes[1])
    Cr_blocks = __unpack_blocks(result[2], quantization_matrixes[1])

    Y = __make_component((result_shape[0], result_shape[1]), Y_blocks)
    Cb = __make_component((result_shape[0]//2, result_shape[1]//2), Cb_blocks)
    Cr = __make_component((result_shape[0]//2, result_shape[1]//2), Cr_blocks)

    Cb = upsampling(Cb)
    Cr = upsampling(Cr)

    YCbCr = np.stack([Y, Cb, Cr], axis = 2)
    rgb = ycbcr2rgb(YCbCr)


    return np.clip(rgb, 0, 255).astype('uint8')


def jpeg_visualize():
    plt.clf()
    img = imread('Lenna.png')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        # Your code here
        y_matrix = own_quantization_matrix(y_quantization_matrix, p)
        color_matrix = own_quantization_matrix(color_quantization_matrix, p)
        compressed = jpeg_compression(img, [y_matrix, color_matrix])
        decompressed = jpeg_decompression(compressed, img.shape, [y_matrix, color_matrix])
            
        axes[i // 3, i % 3].imshow(decompressed)
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg'; 
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    if c_type.lower() == 'jpeg':
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]
        
        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
    elif c_type.lower() == 'pca':
        compressed = []
        for j in range(0, 3):
            compressed.append((pca_compression(img[:, :, j].astype(np.float64).copy(), param)))
            
        img = pca_decompression(compressed)
        compressed.extend([np.mean(img[:, :, 0], axis=1), np.mean(img[:, :, 1], axis=1), np.mean(img[:, :, 2], axis=1)])
        
    if 'tmp' not in os.listdir() or not os.path.isdir('tmp'):
        os.mkdir('tmp')
        
    np.savez_compressed(os.path.join('tmp', 'tmp.npz'), np.array(compressed, dtype = np.object_))
    size = os.stat(os.path.join('tmp', 'tmp.npz')).st_size * 8
    os.remove(os.path.join('tmp', 'tmp.npz'))
        
    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]
    
    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))
     
    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)
    
    ax1.set_title('PSNR for {}'.format(c_type.upper()))
    ax1.plot(param_list, psnr, 'tab:orange')
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('PSNR')
    
    ax2.set_title('Rate-Distortion for {}'.format(c_type.upper()))
    ax2.plot(psnr, rate, 'tab:red')
    ax2.set_xlabel('Distortion')
    ax2.set_ylabel('Rate')
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'pca', [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'jpeg', [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")
