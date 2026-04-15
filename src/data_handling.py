import cv2

def read_image(file_path):
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Cannot find path.")
    return img

def resize_and_crop(img, target_width=800, target_height=600, crop_margin=50):
    """
    Resize x Crop
    """
    img_resized = cv2.resize(img, (target_width, target_height))
    
    img_cropped = img_resized[crop_margin : target_height - crop_margin, 
                              crop_margin : target_width - crop_margin]
    return img_cropped

def convert_to_grayscale(img):
    """
    Convert to grayscale
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def balance_brightness(img_gray, clip_limit=2.0, grid_size=(8, 8)):
    """
    (Contrast Limited Adaptive Histogram Equalization).
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(img_gray)

def reduce_noise_gaussian(img_gray, kernel_size=(5, 5), sigma_x=0):
    return cv2.GaussianBlur(img_gray, kernel_size, sigmaX=sigma_x)