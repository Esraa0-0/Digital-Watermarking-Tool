import PySimpleGUI as sg
import os, io, cv2, tempfile
from algorithms import lsb_embed, lsb_extract, dct_embed, dct_extract, dwt_embed, dwt_extract, psnr, ssim, ber, nc
from PIL import Image
import numpy as np

global original_image 
global secret_message
global stego_image
global extracted_message
global extracted_message2
global embedded_blocks
global attack_img

# Function to convert the image into a format that can be used in the sg.Image element
def convert_to_bytes(image_path):
    image = Image.open(image_path)
    with io.BytesIO() as byte_io:
        image.save(byte_io, format="PNG")  # Save it in PNG format for consistency
        byte_data = byte_io.getvalue()
    return byte_data

orginalImg = [
    [sg.Image(key="-ORIGINAL_IMG-", background_color="white")],
    [sg.Text("Original Image", size=(20, 1), justification="center", text_color="black", background_color="white")]
]

stegoImg = [
    [sg.Image(key="-STEGO_IMG-", background_color="white")], 
    [sg.Text("Stego Image", size=(20, 1), justification="center", text_color="black", background_color="white")]
]

attackImg = [
    [sg.Image(key="-ATTACK_IMG-", background_color="white")], 
    [sg.Text("Image After Attack", size=(20, 1), justification="center", text_color="black", background_color="white")]
]

images = [
    [sg.Column(orginalImg, background_color="white"), 
    sg.Column(stegoImg, background_color="white"),
    sg.Column(attackImg, background_color="white", key='-ATTACK_FIELD-', visible=False)] 
]

attacks = [
    [sg.Radio("None", "ATTACK", key="-NONE-", text_color="black", background_color="white", default=True)],
    [sg.Radio("JPEG Compression", "ATTACK", key="-COMPRESSION-", text_color="black", background_color="white")],
    [sg.Radio("Blurring", "ATTACK", key="-BLUR-", text_color="black", background_color="white")],
    [sg.Radio("Salt & Pepper Noise", "ATTACK", key="-NOISE-", text_color="black", background_color="white")],
    [sg.Radio("Rotation (20°)", "ATTACK", key="-ROTATION-", text_color="black", background_color="white")],
    [sg.Radio("Translation (10px)", "ATTACK", key="-TRANSLATION-", text_color="black", background_color="white")],
    [sg.Button("Apply Attack", key="-APPLY_ATTACK-", size=(12, 1))],
]

extraction = [
    [sg.Text("Extracted Message:", size=(20, 1), font=("Arial bold", 16), text_color="black", background_color="white"), sg.Text("", key="-EXTRACTED_Msg-", text_color="black", background_color="white")],
    [sg.Text("PSNR: ", size=(10, 1), text_color="black", background_color="white"), sg.Text("-", key="-PSNR-", text_color="black", background_color="white")],
    [sg.Text("SSIM: ", size=(10, 1), text_color="black", background_color="white"), sg.Text("-", key="-SSIM-", text_color="black", background_color="white")],
    [sg.Text("BER: ", size=(10, 1), text_color="black", background_color="white"), sg.Text("-", key="-BER-", text_color="black", background_color="white")],
    [sg.Text("NC: ", size=(10, 1), text_color="black", background_color="white"), sg.Text("-", key="-NC-", text_color="black", background_color="white")],
    [sg.Button("Extract Watermark", key="-EXTRACT-", size=(15, 1))]
]

layout = [
    [sg.Text("Watermarking GUI", font=("Arial bold", 20), justification="center", text_color="black", background_color="white", expand_x=True)],
    
    [sg.Text("Watermark (secret Message):", font=("Arial bold", 16),text_color="black", background_color="white"), 
     sg.InputText(key="-WATERMARK-", size=(40, 1)), 
     sg.InputText(key="-UPLOAD-", enable_events=True, visible=False),
     sg.FileBrowse("Upload Image", file_types=(("Image Files", "*.jpg;*.jpeg;*.png;*.bmp"),), size=(12, 1))],

    [sg.Radio("Spatial (LSB)", "METHOD", key="-LSB-", text_color="black", background_color="white", default=True),
     sg.Radio("DCT", "METHOD", key="-DCT-", text_color="black", background_color="white"),
     sg.Radio("DWT", "METHOD", key="-DWT-", text_color="black", background_color="white"),
     sg.Radio("FFT", "METHOD", key="-FFT-", text_color="black", background_color="white"),
     sg.Radio("SVD", "METHOD", key="-SVD-", text_color="black", background_color="white"),
     sg.Button("Embed Watermark", key="-EMBED-", size=(15, 1))],

    [sg.Text("Apply Attacks:", font=("Arial Bold", 16), text_color="black", background_color="white"),
     sg.Column(images, key="-IMAGES-", background_color="white", visible=False)],
    
    [sg.Column(attacks, background_color="white"),
     sg.VSeperator(),
     sg.Column(extraction, background_color="white")]
]

window = sg.Window("Watermarking GUI", layout, button_color=("black", "white"), font=("Arial", 14), background_color="white" ,resizable=True)

while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED:
            break
        
        if  event == "-UPLOAD-": 
            filepath = values["-UPLOAD-"]
            if filepath and os.path.isfile(filepath): 
                original_image = cv2.imread(filepath)
                shape = (original_image.shape[1], original_image.shape[0])
                img_bytes = convert_to_bytes(filepath)
                window["-ORIGINAL_IMG-"].update(data=img_bytes) 
       
        if event == "-EMBED-":
            secret_message = values["-WATERMARK-"]
            if secret_message and values["-UPLOAD-"]:
                
                if values["-LSB-"]:
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)                  
                    original_image = cv2.resize(original_image, (128,128))
                    stego_image, embedded_blocks = lsb_embed(original_image, secret_message)
                    stego_image_resized = cv2.resize(stego_image, shape)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                        temp_file_path = temp_file.name
                        pil_image = Image.fromarray(stego_image_resized)
                        pil_image.save(temp_file_path)
                        window["-STEGO_IMG-"].update(filename=temp_file_path)

                elif values["-DCT-"]:
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)                  
                    original_image = cv2.resize(original_image, (128,128))
                    stego_image = dct_embed(original_image, secret_message)
                    stego_image_resized = cv2.resize(stego_image, shape)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                        temp_file_path = temp_file.name
                        pil_image = Image.fromarray(stego_image_resized)
                        pil_image.save(temp_file_path)
                        window["-STEGO_IMG-"].update(filename=temp_file_path)
                                        
                elif values["-DWT-"]:
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)                  
                    original_image = cv2.resize(original_image, (128,128))
                    stego_image = dwt_embed(original_image, secret_message)
                    stego_image_resized = cv2.resize(stego_image, shape)
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                        temp_file_path = temp_file.name
                        pil_image = Image.fromarray(stego_image_resized)
                        pil_image.save(temp_file_path)
                        window["-STEGO_IMG-"].update(filename=temp_file_path)
                    
                elif values["-FFT-"]:
                    sg.popup("FFT method selected.", background_color="white", text_color="black")
                    
                elif values["-SVD-"]:
                    sg.popup("SVD method selected.", background_color="white", text_color="black")
                    
                attack_img = stego_image.copy()
                window["-IMAGES-"].update(visible=True)

        if event == "-APPLY_ATTACK-":
            if 'original_image' in globals() and 'stego_image' in globals():                
                if values["-NONE-"]: 
                    attack_img = stego_image.copy()
                    window["-ATTACK_IMG-"].update(filename=temp_file_path) 
                    
                elif values["-COMPRESSION-"]: 
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file2:
                        temp_file_path2 = temp_file2.name        
                        quality = 85  # Lower quality value results in higher compression        
                        cv2.imwrite(temp_file_path2, stego_image_resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                        attack_img = cv2.imread(temp_file_path2)          
                        attack_img = cv2.cvtColor(attack_img, cv2.COLOR_BGR2GRAY)                  
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file3:
                            temp_file_path3 = temp_file3.name
                            cv2.imwrite(temp_file_path3, attack_img)     
                            attack_img = cv2.imread(temp_file_path3)      
                            attack_img = cv2.cvtColor(attack_img, cv2.COLOR_BGR2GRAY)                  
                            pil_image = Image.open(temp_file_path3)
                            pil_image.save(temp_file_path3) 
                            window["-ATTACK_IMG-"].update(filename=temp_file_path3)
                                   
                elif values["-BLUR-"]:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file2:
                        temp_file_path2 = temp_file2.name       
                        attack_img = cv2.GaussianBlur(stego_image_resized, (15, 15), 0)  # Kernel size (15, 15)        
                        pil_image2 = Image.fromarray(attack_img)
                        pil_image2.save(temp_file_path2)       
                        window["-ATTACK_IMG-"].update(filename=temp_file_path2)
                    
                elif values["-NOISE-"]:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file2:
                        temp_file_path2 = temp_file2.name       
                        prob = 0.05  # Probability of noise (adjustable)        
                        attack_img = stego_image_resized.copy()  
                        num_salt = int(prob * attack_img.size)  # Number of salt (white) pixels
                        salt_coords = [np.random.randint(0, i, num_salt) for i in attack_img.shape]  # Generate random coordinates
                        for x, y in zip(salt_coords[0], salt_coords[1]):
                            attack_img[x, y] = 255  # Salt (white)
                        num_pepper = int(prob * attack_img.size)  # Number of pepper (black) pixels
                        pepper_coords = [np.random.randint(0, i, num_pepper) for i in attack_img.shape]  # Generate random coordinates
                        for x, y in zip(pepper_coords[0], pepper_coords[1]):
                            attack_img[x, y] = 0  # Pepper (black)
                        pil_image2 = Image.fromarray(attack_img)
                        pil_image2.save(temp_file_path2)       
                        window["-ATTACK_IMG-"].update(filename=temp_file_path2)
                    
                elif values["-ROTATION-"]:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file2:
                        temp_file_path2 = temp_file2.name
                        angle = 20  # Rotation angle
                        center = (stego_image_resized.shape[1] // 2, stego_image_resized.shape[0] // 2)  # Center of the image
                        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # The '1.0' scale factor keeps the size
                        attack_img = cv2.warpAffine(stego_image_resized, rotation_matrix, (stego_image_resized.shape[1], stego_image_resized.shape[0]))
                        pil_image2 = Image.fromarray(attack_img)
                        pil_image2.save(temp_file_path2)
                        window["-ATTACK_IMG-"].update(filename=temp_file_path2)

                elif values["-TRANSLATION-"]:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file2:
                        temp_file_path2 = temp_file2.name
                        tx = 10  # Translate 10 pixels in the x direction
                        ty = 10  # Translate 10 pixels in the y direction
                        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
                        attack_img = cv2.warpAffine(stego_image_resized, translation_matrix, (stego_image_resized.shape[1], stego_image_resized.shape[0]))
                        pil_image2 = Image.fromarray(attack_img)
                        pil_image2.save(temp_file_path2)
                        window["-ATTACK_IMG-"].update(filename=temp_file_path2)
                    
                window["-ATTACK_FIELD-"].Update(visible=True)
                
        if event == "-EXTRACT-":
            if 'original_image' in globals() and 'stego_image' in globals(): 
                if values["-LSB-"]:
                    window["-EXTRACTED_Msg-"].update(lsb_extract(attack_img, embedded_blocks))
                    extracted_message = window["-EXTRACTED_Msg-"]
                    extracted_message2 = lsb_extract(attack_img, embedded_blocks)
                    
                elif values["-DCT-"]:
                    window["-EXTRACTED_Msg-"].update(dct_extract(attack_img))
                    extracted_message = window["-EXTRACTED_Msg-"]
                    extracted_message2 = dct_extract(attack_img)
                    
                elif values["-DWT-"]:
                    window["-EXTRACTED_Msg-"].update(dwt_extract(attack_img))
                    extracted_message = window["-EXTRACTED_Msg-"]
                    extracted_message2 = dwt_extract(attack_img)
                    
                elif values["-FFT-"]:
                    sg.popup("FFT method selected.", background_color="white", text_color="black") 
                       
                elif values["-SVD-"]:
                    sg.popup("SVD method selected.", background_color="white", text_color="black")
                    
                if (values["-NONE-"]):    
                    attack_img = stego_image
                else:
                   attack_img = cv2.resize(attack_img, (128, 128))
                   
                window["-PSNR-"].update(psnr(original_image, stego_image))
                window["-SSIM-"].update(ssim(original_image, stego_image))
                window["-BER-"].update(ber(secret_message, extracted_message2))
                window["-NC-"].update(nc(secret_message, extracted_message2))
            
window.close()
