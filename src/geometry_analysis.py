import cv2
import numpy as np
import math


def detect_edges(img_gray, low_thresh=60, high_thresh=150):
    edges = cv2.Canny(img_gray, low_thresh, high_thresh)
    return edges

def get_lines (edges_img, threshold=80, minLineLength=110, maxLineGap=80):
    return cv2.HoughLinesP(
        edges_img, 
        1, 
        np.pi/180, 
        threshold=threshold, 
        minLineLength=minLineLength, 
        maxLineGap=maxLineGap
    )


def filter_lines_by_length(lines, min_len=60, max_len=600):
    """
    Lọc đường thẳng dựa trên độ dài (Euclidean distance).
    Loại bỏ các đường quá ngắn (nhiễu) hoặc quá dài (vết nứt ngang đường).
    """
    if lines is None:
        return None

    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        if min_len <= length <= max_len:
            filtered_lines.append(line)
            
    return filtered_lines

def apply_roi_mask(img, upper_limit=0.4):
    """
    Che đi phần trên của ảnh.
    upper_limit: Tỷ lệ phần trên muốn che (0.5 = 50%).
    """
    mask = np.zeros_like(img)
    height, width = img.shape[:2]
    
    start_y = int(height * upper_limit)
    
    polygons = np.array([
        [(0, height), (width, height), (width, start_y), (0, start_y)]
    ])
    
    if len(img.shape) > 2: # Ảnh màu
        cv2.fillPoly(mask, polygons, (255, 255, 255))
    else: # Ảnh xám/biên
        cv2.fillPoly(mask, polygons, 255)
        
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def filter_lines_by_angle(lines, min_angle=5, max_angle=80):
    """
    Lọc các đường thẳng dựa trên góc nghiêng (độ).
    min_angle, max_angle: Khoảng góc muốn giữ lại (ví dụ 20 đến 80 độ).
    """
    filtered_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            angle_rad = math.atan2(abs(y2 - y1), abs(x2 - x1))
            angle_deg = math.degrees(angle_rad)
            
            if min_angle <= angle_deg <= max_angle:
                filtered_lines.append(line)
                
    return filtered_lines