from imports import *


def RGB_to_Grayscale(img: np.ndarray) -> np.ndarray:
    img_grayscale = 0.3 * img[:, :, 0] + 0.6 * img[:, :, 1] + 0.1 * img[:, :, 2]
    img_grayscale = img_grayscale.astype('uint8')
    return img_grayscale


def grayscale_to_binary(img_gray: np.array, threshold: int) -> np.array:
    img_bin = np.where(img_gray >= threshold, 1, 0).astype(np.uint8)
    return img_bin


def image_to_array(path: str, show: bool) -> np.array:
    img = plt.imread(path)
    if show:
        plt.figure()
        if img.ndim < 3:
            plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        else:
            plt.imshow(img, vmin=0, vmax=255)
        plt.show()
    return img


def standardize_image(img: np.ndarray):
    # inteleg ca ar trebui scazuta media si api scazuta abaterea standard
    std = np.std(img)
    mean = np.mean(img)
    standardized_img = (img - mean) / std
    return standardized_img


def Normalize_image(img: np.ndarray) -> np.ndarray:
    norm_img = (img - 128) / 127
    return norm_img


def padding(img: np.ndarray, padding_size: int, padding_type) -> np.ndarray:
    # matricea va fi de (n+2padding_size)*(m+2padding_size)
    n = img.shape[0]
    m = img.shape[1]
    if m < padding_size or n < padding_size:
        warnings.simplefilter("padding_size unacceptable. pedding_zize must be less than " + str(min(n, m)))
        return img
    if padding_type == "constant":
        padded_img = np.zeros((n + 2 * padding_size, m + 2 * padding_size))
        padded_img[padding_size:padding_size + n, padding_size:padding_size + m] += img
    elif padding_type == "mirror":
        # colturile patratului vor fi numeroatate: r-lrft and up-down
        A1 = img[0:padding_size, 0:padding_size]
        A2 = img[0:padding_size, m - padding_size:m]
        A3 = img[n - padding_size:n, 0:padding_size]
        A4 = img[n - padding_size:n, m - padding_size:m]
        padded_img = np.zeros((n + 2 * padding_size, m + 2 * padding_size))
        p = padding_size  # ma dor ochii
        # latura stanga
        padded_img[p:n + p, 0:p] += np.fliplr(img[:, 0:p])
        # latura dreapta
        padded_img[p:n + p, m + p:(m + 2 * p)] += np.fliplr(img[:, m - p:m])
        # latura superioara
        padded_img[0:p, p:m + p] += np.flipud(img[0:p, :])
        # latura inferioara
        padded_img[n + p:n + 2 * p, p:m + p] += np.flipud(img[n - p:n, :])
        # se adauga patratele:
        padded_img[0:p, 0:p] += np.flipud(np.fliplr((A1)))
        padded_img[0:p, m + p:m + 2 * p] += np.flipud(np.fliplr((A2)))
        padded_img[n + p:n + 2 * p, 0:p] += np.flipud(np.fliplr((A3)))
        padded_img[n + p:n + 2 * p, m + p:m + 2 * p] += np.flipud(np.fliplr((A4)))
        # se adauga si centrul:
        padded_img[padding_size:padding_size + n, padding_size:padding_size + m] += img
    else:
        print("NEIMPLEMENTAT")
        padded_img = img
    return padded_img


def norm_image(img: np.array) -> np.array:
    img_normed = img / 255
    return img_normed


def pooling(list_of_patches: list, type_of_pooling: str) -> np.ndarray:
    if type_of_pooling == "max":
        returned_list = np.array(list(map(np.max, list_of_patches)))
    elif type_of_pooling == "min":
        returned_list = np.array(list(map(np.min, list_of_patches)))
    elif type_of_pooling == "mean":
        returned_list = np.array((np.uint8(list(map(np.mean, list_of_patches)))))
    else:
        print("NEIMPLEMENTAT: type_of_pooling must be: max or mean or min")

    return returned_list


def histogram(img: np.array, plot: bool = False) -> np.array:
    hist, bin_edges = np.histogram(img, bins=256, range=(0, 256), density=True)
    if plot:
        plt.figure()
        plt.bar(bin_edges[:-1], hist)
        plt.show()
    return hist


def restoration_metrics(original_image: np.array, reconstructed_image: np.array):
    module_diff = np.abs(reconstructed_image - original_image)
    SNR = 10 * np.log10(np.sum(original_image ** 2) / np.sum(module_diff ** 2))
    MSE = np.mean(module_diff ** 2)
    MAE = np.mean(module_diff)
    PSNR = 10 * np.log10(np.max(original_image) ** 2 / MSE)
    return MSE, MAE, SNR, PSNR


def patching(img :np.ndarray, patch_size :tuple, stride :tuple) -> np.ndarray:
    lin, col, depth = img.shape
    img_pad = np.zeros((lin + patch_size[0], col + patch_size[1], depth))
    img_pad[:lin, :col, :] = img.copy()
    patches = []
    for i in range(0, lin, stride[0]):
        for j in range(0, col, stride[1]):
            patches.append(img_pad[i: i+patch_size[0], j: j+patch_size[1], :])
    return np.array(patches)


def restore_image(list_of_patches, stride, img_size, overlap=0):
    lin, col, depth = img_size
    patch_size = list_of_patches[0].shape
    img_restored = np.zeros((lin+patch_size[0], col+patch_size[1], depth))

    idx = 0
    # Reconstruim poza din patch-uri fara mediere
    for i in range(0, lin, stride[0]):
        for j in range(0, col, stride[1]):
            overlaped_patch = list_of_patches[idx][int(overlap * patch_size[0]):, int(overlap * patch_size[1]):, :]
            non_overlaped_patch = list_of_patches[idx][: int(overlap * patch_size[0]), : int(overlap * patch_size[1]), :]
            img_restored[i: i+int(overlap*patch_size[0]), j: j+int(overlap*patch_size[1]), :] = non_overlaped_patch
            img_restored[i + int(overlap * patch_size[0]): i + patch_size[0], j + int(overlap * patch_size[1]): j + patch_size[1], :] = overlaped_patch
            idx += 1


    idx = 0
    # Mediem peste zonele de overlapping
    for i in range(0, lin, stride[0]):
        for j in range(0, col, stride[1]):
            overlaped_patch = list_of_patches[idx][int(overlap * patch_size[0]):, int(overlap * patch_size[1]):, :]
            non_overlaped_patch = list_of_patches[idx][: int(overlap * patch_size[0]), : int(overlap * patch_size[1]), :]
            overlaped_img = img_restored[i: i+int(overlap*patch_size[0]), j: j+int(overlap*patch_size[1]), :]
            non_overlaped_img = img_restored[i+int(overlap*patch_size[0]): i+patch_size[0], j+int(overlap*patch_size[1]): j+patch_size[1], :]
            img_restored[i: i+int(overlap*patch_size[0]), j: j+int(overlap*patch_size[1]), :] = np.mean([overlaped_img, non_overlaped_patch], axis=0)
            img_restored[i + int(overlap * patch_size[0]): i + patch_size[0], j + int(overlap * patch_size[1]): j + patch_size[1], :] = overlaped_patch #np.mean([non_overlaped_img, overlaped_patch], axis=0)
            idx += 1

    return img_restored[:lin, :col, :]


def upsampling(list_of_pool_values: list, upsampling_shape: tuple) -> np.array:
    list = []
    for i in list_of_pool_values:
        list.append(np.full(upsampling_shape, i))
    return np.array(list)


def heatmap(img_org: np.array, img_rec: np.array, mode: str, plot: bool) -> np.array:
    heat_map = np.empty(shape=img_org.shape)
    if mode == "abs":
        heat_map = np.abs(img_org - img_rec)
    elif mode == "squared":
        heat_map = (img_org - img_rec) ** 2
    if plot:
        plt.figure()
        plt.imshow(heat_map, interpolation="none")
        plt.colorbar()
        plt.show()
    return heat_map


def pipeline (path: str, params: dict):
    '''
    Wiki_pipeline
        Dictionary of the keys in the params dictionary:
        patch_size = Number of patches for the PATCHING function;
        stride = Stride size for the PATCHING function;
        type_of_pooling = type argument of the POOLING function. It can be: max, min, mean;
        upsampling_shape = tuple argument for UPSAMPLING function;
        stride_restored_image = Stride size for RESTORE IMAGE function;
        img_size = tuple argument of M and N, the size of the image;
        heatmap_mode = mode of heatmap: abs or squared
        heatmap_plot = bool value, if it's 1 will plot the error heat map
    '''
    img = image_to_array(path, False)
    if img.ndim == 3:
        img = RGB_to_Grayscale(img)
    img_copy = img.copy()
    img = patching (img, params.get('patch_size'), params.get('stride'))
    pool_list = pooling(img, params.get('type_of_pooling'))
    upsamp = upsampling(pool_list, params.get('upsampling_shape'))
    restored_image = restore_image (upsamp, params.get('stride_restore_image'), img_copy.shape)
    MSE, MAE, SNR, PSNR = restoration_metrics(img_copy, restored_image)
    heat_map = heatmap(img_copy, restored_image, params.get('heatmap_mode'), params.get('heatmap_plot'))
    return MSE, MAE, SNR, PSNR, heat_map


def conv(img, kernel, stride, padding_mode, padding_type):
    """
    Function that implements the 2D convolution algorithm

    Parameters
    ----------
    img : numpy 2D array
        Image to convolve on.
    kernel : numpy 2D array
        Kernel to be used in convolution.
    stride : tuple
        Step size for each dimension.
    padding_mode : str, can be "valid" or "same"
        If padding is "same", the convoled image will have the same shape as the original image.
        If padding is "valid", the convoled image will shrink according to kernel_size.
    padding_type : str, can be "mirror" or "constant".
        If "mirror", the image border will be mirrored.
        If "constant", the image will be padded with zeros.

    Returns
    -------
    img_conv : numpy 2D array
        Convoled image.

    """
    lin, col = img.shape
    kernel_size = kernel.shape[0] // 2

    if padding_mode == 'same':
        img_conv = padding(img, kernel_size, padding_type)
    else:
        img_conv = img.copy()

    for i in range(kernel_size, lin - kernel_size, stride[0]):
        for j in range(kernel_size, col - kernel_size, stride[1]):
            selected = img[i - kernel_size: i + kernel_size + 1, j - kernel_size: j + kernel_size + 1]
            img_conv[i, j] = np.abs(np.sum(selected * kernel))

    if padding_mode == 'same':
        lin, col = img_conv.shape
    img_conv = img_conv[kernel_size:lin - kernel_size, kernel_size:col - kernel_size]
    return np.uint8(img_conv)


class Generator:

    def __init__(self, train: str, load_path: str, save_path: str, stride: tuple, patch_size: tuple, train_test_split_percentage: float):
        self.load_path = load_path
        self.save_path = save_path
        self.stride = stride
        self.patch_size = patch_size
        self.train_test_split_percentage = train_test_split_percentage
        self.train = train

    def load_data(self, show=False):
        """
        This method takes every image from the path we give to it via load_path, making patches of it and
        save the patches as a tuple of (GT_patch, Noisy_patch) in a .npy file.
        :param show:
        Don't use it if you don't want to see the images which are transformed into array. It's a parameter of
        image_to_array function.
        :return:
        patch_shape: It's useful for the first layer of the NN.
        """
        files = [os.path.join(self.load_path, file) for file in os.listdir(self.load_path)]
        image_counter = 0
        patch_shape = 0
        for i in range(len(files)):
            images = [os.path.join(files[i], image) for image in (os.listdir(files[i]))]
            img = image_to_array(images[0], show)
            patched_img = patching(img, self.patch_size, self.stride)
            noisy = image_to_array(images[1], show)
            patched_noisy = patching(noisy, self.patch_size, self.stride)
            for patch1, patch2, j in zip(patched_noisy, patched_img, range(len(patched_img))):
                patch_shape = np.shape(patch1)
                if (image_counter+1) % (1/self.train_test_split_percentage) != 0:
                    # np.save(os.path.join(self.save_path, 'TRAIN_IMG_pic{0}.npy'.format(image_counter)),
                    #         np.array([patch1, patch1 - patch2])) # Pentru DnCNN -> GT - Zgomot
                  np.save(os.path.join(self.save_path, 'TRAIN_IMG_pic{0}.npy'.format(image_counter)),
                          np.array([patch1, patch2])) # Pentru varianta in care ii dam GT - NOISY

                else:
                    # np.save(os.path.join(self.save_path, 'VALIDATION_IMG_pic{0}.npy'.format(image_counter)),
                    #         np.array([patch1, patch1 - patch2])) # Pentru DnCNN -> GT - ZGOMOT
                    np.save(os.path.join(self.save_path, 'VALIDATION_IMG_pic{0}.npy'.format(image_counter)),
                            np.array([patch1, patch2])) # Pentru varianta clasica, GT - NOISY
                image_counter += 1
        return patch_shape


    def __call__ (self):
        """
        :param train_or_validation: 'train' or 'validation'.
         Takes the decision if the patch is used for train or validation because they're splitted already.
        :return:
        Feeding the NN with patches.
        """
        if self.train == 'train':
            patches = [os.path.join(self.save_path, patch) for patch in os.listdir(self.save_path)
                       if patch.startswith("TRAIN")]
        elif self.train == 'validation':
            patches = [os.path.join(self.save_path, patch) for patch in os.listdir(self.save_path)
                       if patch.startswith("VALIDATION")]
        for i in range (len(patches)):
            x = np.load(patches[i])
            yield x[0], x[1]

"""
Warnings:
    1. __call__ method is not making batches of patches so we have to be careful with the batches.
    2. The shape of a patch is the return of the load_data method.
"""


def project_onto_hyperslab (point: tuple, delta: float):
    d = int(point[0]) - int(point[1])
    u_new = point[0]
    v_new = point[1]
    if d > delta:
        u_new = (int(point[0]) + int(point[1])) / 2 + delta/2
        v_new = (int(point[0]) + int(point[1])) / 2 - delta/2
    elif d < -delta:
        u_new = (int(point[0]) + int(point[1])) / 2 - delta/2
        v_new = (int(point[0]) + int(point[1])) / 2 + delta/2
    return np.uint8(u_new), np.uint8(v_new)



def project_onto_c1(P :tuple, Q :tuple, x :np.ndarray, parity :bool, delta :float) ->np.ndarray:

    def justdoit(start1, start2):
        for line_index in range(Q1, min(m, Q2), 1):
            for i, j in zip(range(start1, min(n-1, P2), 2), range(start2, min(n, P2), 2)):
                tuplu = (yp[line_index, i],yp[line_index, j])
                yp[line_index, i], yp[line_index, j] = np.uint8(project_onto_hyperslab(tuplu ,delta))
        return yp

    P1, P2 = P
    Q1, Q2 = Q
    m, n = np.shape(x)
    yp = np.copy(x)
    if parity:
        return justdoit(P1, P1+1)[Q1:Q2, P1:P2]
    else:
        return justdoit(P1+1, P1+2)[Q1:Q2, P1:P2]

def project_onto_c2 (P :tuple, Q :tuple, x :np.ndarray, parity :bool, delta :float) ->np.ndarray:

    def justdoit(start1, start2):
        for col_index in range(P1, min(n, P2), 1):
            for i, j in zip(range(start1, min(m-1, Q2), 2), range(start2, min(m, Q2), 2)):
                yp[i, col_index], yp[j, col_index] = np.uint8(project_onto_hyperslab((yp[i, col_index], yp[j, col_index]),delta))
        return yp

    P1, P2 = P
    Q1, Q2 = Q
    m, n = np.shape(x)
    yp = np.copy(x)
    if parity:
        return justdoit(Q1, Q1+1)[Q1:Q2, P1:P2]
    else:
        return justdoit(Q1+1, Q1+2)[Q1:Q2, P1:P2]

def project_onto_c3 (P :tuple, Q :tuple, x :np.ndarray, parity :bool, delta :float) ->np.ndarray:
    P1, P2 = P
    Q1, Q2 = Q
    yp = np.copy(x)
    if parity:
        for i, j in zip (range(P1, P2-1, 2), range(Q1, Q2-1, 2)):
            yp [i,j], yp[i+1, j+1] = np.uint8 (project_onto_hyperslab((yp[i,j], yp[i+1, j+1]), delta))
    else:
        for i, j in zip (range(P1+1, P2, 2), range(Q1+1, Q2-1, 2)):
            yp [i,j], yp[i+1, j+1] = np.uint8 (project_onto_hyperslab((yp[i,j], yp[i+1, j+1]), delta))

    return yp


def POCS (y: np.array, x: np.array, P: tuple, Q: tuple, delta: float, max_iter: int, precision: float, m = 6):
    """
    y : ground truth image
    x : noisy image
    P : tuple with the horizontal lower and upper bound of the noisy area
    Q : tuple with the vertical lower and upper bound of the noisy area
    delta : minimum distance between two adjacent pixels
    max_iter : maximum number of iterations
    precision : stopping condition
    """

    no_iterations = 0
    yp = np.copy(x)
    error = 10^4

    while True:

        if error < precision:
            break
        if no_iterations >= max_iter:
            break

        no_iterations += 1
        reminder = no_iterations % m
        print(no_iterations,error)
        if reminder == 0:
            yp[Q[0]:Q[1], P[0]:P[1]] = project_onto_c1(P = P, Q = Q, x = x, parity = False, delta = delta)
        elif  reminder == 1:
            yp[Q[0]:Q[1], P[0]:P[1]] = project_onto_c1(P = P, Q = Q, x = x, parity = True, delta = delta)
        elif reminder == 2:
            yp[Q[0]:Q[1], P[0]:P[1]] = project_onto_c2(P = P, Q = Q, x = x, parity = False, delta = delta)
        elif reminder == 3:
            yp[Q[0]:Q[1], P[0]:P[1]]= project_onto_c2(P = P, Q = Q, x = x, parity = True, delta = delta)
        if reminder == 4:
            yp= project_onto_c3(P = P, Q = Q, x = x, parity = False, delta = delta)
        elif reminder == 5:
            yp = project_onto_c3(P = P, Q = Q, x = x, parity = True, delta = delta)

        #error = np.sum(sum(abs(a - b) for a, b in zip(y, yp)))
        error = np.sum(np.abs(y-yp))


    return yp, error




