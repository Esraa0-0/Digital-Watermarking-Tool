import numpy as np
from skimage.metrics import structural_similarity as calculate_ssim 
from scipy.fftpack import dctn, idctn
import pywt

# Function to embed secret message in an image using LSB
def lsb_embed(image, message, block_size=8):
    binary_message = ''.join(format(ord(char), '08b') for char in message) + '11111111'  # End delimiter
    message_len = len(binary_message)
    message_index = 0

    stego_image = image.copy()
    height, width = stego_image.shape
    embedded_blocks = []
    
    # Calculate standard deviation for each block and store with coordinates
    block_std_devs = []
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i + block_size, j:j + block_size]
            std_dev = np.std(block)
            block_std_devs.append((std_dev, i, j))

    # Sort blocks by standard deviation in ascending order
    block_std_devs.sort()
    
    # Embed message in the blocks with the lowest standard deviations
    for std_dev, i, j in block_std_devs:
        if message_index >= message_len:
            break
        embedded_blocks.append((i, j))
        for x in range(block_size):
            for y in range(block_size):
                if message_index < message_len:
                    pixel = stego_image[i + x, j + y]
                    stego_image[i + x, j + y] = (pixel & ~1) | int(binary_message[message_index])
                    message_index += 1
    
    return stego_image, embedded_blocks

# Function to extract secret message from a stegano image using LSB
def lsb_extract(image, blocks, block_size=8):
    extracted_bits = []
    for (i, j) in blocks:
        for x in range(block_size):
            for y in range(block_size):
                extracted_bits.append(image[i + x, j + y] & 1)
                
    binary_data = ''.join(map(str, extracted_bits))
    all_bytes = [binary_data[i:i + 8] for i in range(0, len(binary_data), 8)]
    
    extracted_message = ""
    for byte in all_bytes:
        if byte == "11111111":  # End delimiter
            break
        extracted_message += chr(int(byte, 2))
        
    return extracted_message

def dct_embed(image, watermark, delta=10):
    binary_watermark = ''.join(format(ord(char), '08b') for char in watermark) + '11111111'
    watermark_len = len(binary_watermark)
    watermark_index = 0

    watermarked_image = image.copy()
    height, width = watermarked_image.shape

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            if watermark_index >= watermark_len:
                break
            block = watermarked_image[i:i + 8, j:j + 8]

            dct_block = dctn(block, norm='ortho')

            bit = binary_watermark[watermark_index]
            if bit == '1':
                dct_block[4, 4] = abs(dct_block[4, 4]) + delta
            else:
                dct_block[4, 4] = -abs(dct_block[4, 4]) - delta

            idct_block = idctn(dct_block, norm='ortho')
            watermarked_image[i:i + 8, j:j + 8] = np.clip(idct_block, 0, 255)

            watermark_index += 1

    return watermarked_image

def dct_extract(watermarked_image):
    extracted_bits = []
    height, width = watermarked_image.shape

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = watermarked_image[i:i + 8, j:j + 8]
            if block.shape != (8, 8):  # Handle edge cases
                continue

            dct_block = dctn(block, norm='ortho')

            coefficient = dct_block[4, 4]
            extracted_bits.append('1' if coefficient > 0 else '0')

    binary_data = ''.join(extracted_bits)
    all_bytes = [binary_data[i:i + 8] for i in range(0, len(binary_data), 8)]

    extracted_watermark = ""
    for byte in all_bytes:
        if len(byte) != 8:  # Skip incomplete bytes
            continue
        char = chr(int(byte, 2))
        if char == '\xFF':  # End delimiter
            break
        extracted_watermark += char

    return extracted_watermark

def dwt_embed(image, watermark, q=10):
    binary_watermark = ''.join(format(ord(char), '08b') for char in watermark) + '11111111'
    watermark_len = len(binary_watermark)
    watermark_index = 0

    watermarked_image = image.copy()
    height, width = watermarked_image.shape

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            if watermark_index >= watermark_len:
                break
            block = watermarked_image[i:i + 8, j:j + 8]

            coeffs2 = pywt.dwt2(block, 'haar')
            LL, (LH, HL, HH) = coeffs2

            bit = binary_watermark[watermark_index]
            mean_LL = np.mean(LL)

            if bit == '1':
                if int(mean_LL / q) % 2 != 1:
                    mean_LL += q
            else:
                if int(mean_LL / q) % 2 != 0:
                    mean_LL += q

            adjustment = mean_LL - np.mean(LL)
            LL += adjustment 

            watermarked_block = pywt.idwt2((LL, (LH, HL, HH)), 'haar')
            watermarked_image[i:i + 8, j:j + 8] = np.clip(watermarked_block, 0, 255)

            watermark_index += 1

    return watermarked_image

def dwt_extract(watermarked_image, q=10):
    extracted_bits = []
    height, width = watermarked_image.shape
    termination_found = False

    for i in range(0, height, 8):
        for j in range(0, width, 8):
            block = watermarked_image[i:i + 8, j:j + 8]

            coeffs2 = pywt.dwt2(block, 'haar')
            LL, (LH, HL, HH) = coeffs2

            mean_LL = np.mean(LL)

            if int(mean_LL / q) % 2 == 1:
                extracted_bits.append('1')
            else:
                extracted_bits.append('0')

            if ''.join(extracted_bits[-8:]) == '11111111':
                termination_found = True
                break

        if termination_found:
            break

    binary_data = ''.join(extracted_bits)
    all_bytes = [binary_data[i:i + 8] for i in range(0, len(binary_data), 8)]

    extracted_watermark = ""
    for byte in all_bytes:
        if len(byte) != 8:  # Skip incomplete bytes
            continue
        char = chr(int(byte, 2))
        if char == '\xFF':  # End delimiter
            break
        extracted_watermark += char

    return extracted_watermark

# Example usage
'''image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)  # Example image
watermark = "Hello"
watermarked_image = dwt_embed(image, watermark)
extracted_watermark = dwt_extract(watermarked_image)
print("Extracted Watermark:", extracted_watermark)'''

# Function to calculate Peak Signal-to-Noise Ratio (PSNR)
def psnr(original_image, encoded_image):
    original_pixels = np.array(original_image)
    encoded_pixels = np.array(encoded_image)
    max_pixel_value=255

    # Calculate MSE between the original and encoded image
    mse = np.mean((original_pixels - encoded_pixels) ** 2)
    if mse == 0:  # No difference between images
        return float('inf')
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return str(psnr)+" db"

# Function to calculate Structural Similarity Index (SSIM)
def ssim(original_image, encoded_image):
    original_gray = np.array(original_image)  # Grayscale already
    encoded_gray = np.array(encoded_image)
    return calculate_ssim(original_gray, encoded_gray)

# Function to calculate Bit Error Rate (BER)
def ber(original_watermark, extracted_watermark):
    # Convert Text objects to strings
    original_text = str(original_watermark)
    extracted_text = str(extracted_watermark)

    # Ensure both texts are the same length by padding the shorter one with spaces
    max_length = max(len(original_text), len(extracted_text))
    original_text = original_text.ljust(max_length)
    extracted_text = extracted_text.ljust(max_length)

    # Convert characters to binary and compare bits
    total_bits = max_length * 8  # Each character has 8 bits (ASCII)
    different_bits = 0

    for orig_char, ext_char in zip(original_text, extracted_text):
        orig_binary = format(ord(orig_char), '08b')  # Convert to binary
        ext_binary = format(ord(ext_char), '08b')    # Convert to binary
        for bit1, bit2 in zip(orig_binary, ext_binary):
            if bit1 != bit2:
                different_bits += 1

    # Calculate Bit Error Rate (BER)
    ber = different_bits / total_bits
    return ber

# Function to calculate Normalized Correlation (NC)
def nc(original_watermark, extracted_watermark):
    # Convert Text objects to strings
    original_text = str(original_watermark)
    extracted_text = str(extracted_watermark)

    # Ensure both texts are the same length by padding the shorter one with spaces
    max_length = max(len(original_text), len(extracted_text))
    original_text = original_text.ljust(max_length)
    extracted_text = extracted_text.ljust(max_length)

    # Convert characters to ASCII values and compute NC
    orig_values = [ord(char) for char in original_text]
    ext_values = [ord(char) for char in extracted_text]

    # Compute the dot product of the two text ASCII values
    dot_product = sum(o * e for o, e in zip(orig_values, ext_values))

    # Compute the Euclidean norms of the original and extracted ASCII values
    norm_original = sum(o ** 2 for o in orig_values) ** 0.5
    norm_extracted = sum(e ** 2 for e in ext_values) ** 0.5

    # Compute Normalized Correlation (NC)
    if (norm_original * norm_extracted)==0:
        return 0
    
    nc = dot_product / (norm_original * norm_extracted)
    return nc
