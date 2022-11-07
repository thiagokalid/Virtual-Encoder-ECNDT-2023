import numpy as np
from scipy import signal

# Utilities associated with digital image processing.

def generate_window(img_dim, window_type=None):
    if window_type is None:
        return np.ones(img_dim)
    elif window_type == 'Blackman-Harris':
        window1dy = signal.windows.blackmanharris(img_dim[0])
        window1dx = signal.windows.blackmanharris(img_dim[1])
        window2d = np.sqrt(np.outer(window1dy, window1dx))
        return window2d
    elif window_type == 'Blackman':
        window1dy = np.abs(np.blackman(img_dim[0]))
        window1dx = np.abs(np.blackman(img_dim[1]))
        window2d = np.sqrt(np.outer(window1dy, window1dx))
        return window2d


def apply_window(img1, img2, window_type):
    window = generate_window(img1.shape, window_type)
    return img1 * window, img2 * window


def autocontrast(img):
    autocontrast_img = img
    input_minval = autocontrast_img.min()
    input_maxval = autocontrast_img.max()
    output_img = (autocontrast_img - input_minval) / (input_maxval - input_minval) * 255
    return output_img


def gaussian_noise(image, snr):
    signal_energy = np.power(np.linalg.norm(image), 2)
    # snr = 20 * log10(signal_energy / noise_energy)
    # Therefore: noise_energy = 10^(- snr/20 + log10(signal_energy)
    noise_energy = np.power(10, np.log10(signal_energy) - snr / 20)

    # noise_energy ^2 =~ M * sigma^2 => sigma = sqrt(noise_energy^2 / M),
    # where sigma is the standard deviation and M is the signal length
    M = image.size  # Total number of elements
    sd = np.sqrt(np.power(noise_energy, 2) / M)
    sd = np.mean(image) * 0.2
    noise_signal = np.random.normal(0, sd, M)
    flattened_image = image.flatten(order="F") + noise_signal
    noisy_image = np.reshape(flattened_image, image.shape, order='F')
    return noisy_image


def salt_and_pepper(image, prob=0.05):
    # If the specified `prob` is negative or zero, we don't need to do anything.
    if prob <= 0:
        return image

    arr = np.asarray(image)
    original_dtype = arr.dtype

    # Derive the number of intensity levels from the array datatype.
    intensity_levels = 2 ** 8

    min_intensity = 0
    max_intensity = intensity_levels - 1

    # Generate an array with the same shape as the image's:
    # Each entry will have:
    # 1 with probability: 1 - prob
    # 0 or np.nan (50% each) with probability: prob
    random_image_arr = np.random.choice(
        [min_intensity, 1, np.nan], p=[prob / 2, 1 - prob, prob / 2], size=arr.shape
    )

    # This results in an image array with the following properties:
    # - With probability 1 - prob: the pixel KEEPS ITS VALUE (it was multiplied by 1)
    # - With probability prob/2: the pixel has value zero (it was multiplied by 0)
    # - With probability prob/2: the pixel has value np.nan (it was multiplied by np.nan)
    # We need to to `arr.astype(np.float)` to make sure np.nan is a valid value.
    salt_and_peppered_arr = arr.astype(np.float) * random_image_arr

    # Since we want SALT instead of NaN, we replace it.
    # We cast the array back to its original dtype so we can pass it to PIL.
    salt_and_peppered_arr = np.nan_to_num(
        salt_and_peppered_arr, nan=max_intensity
    ).astype(original_dtype)

    return salt_and_peppered_arr
