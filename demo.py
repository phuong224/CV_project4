"""
Demo tương tác — CV Project 4
Chạy: python demo.py
Mở browser tại: http://localhost:7860
"""

import cv2
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")  # non-interactive backend cho Gradio
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import gradio as gr

# ---------------------------------------------------------------------------
# Load YOLO model một lần duy nhất khi khởi động
# ---------------------------------------------------------------------------
MODEL_PATH = "results/models/yolov8n.pt"
yolo_model = YOLO(MODEL_PATH)

# ---------------------------------------------------------------------------
# Tiền xử lý (khớp với src/data_handling.py)
# ---------------------------------------------------------------------------
def preprocess_gray(img_bgr, target_width=400, target_height=400, crop_margin=50):
    """Resize, crop, grayscale, CLAHE, medianBlur — khớp pipeline chính."""
    img = cv2.resize(img_bgr, (target_width, target_height))
    img = img[crop_margin:target_height - crop_margin, crop_margin:target_width - crop_margin]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.medianBlur(gray, ksize=5)
    return gray

def preprocess_rgb(img_bgr, target_width=400, target_height=400, crop_margin=50):
    """Resize, crop và trả về ảnh RGB để hiển thị."""
    img = cv2.resize(img_bgr, (target_width, target_height))
    img = img[crop_margin:target_height - crop_margin, crop_margin:target_width - crop_margin]
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------------------------------------------------------------------------
# Phát hiện biên (khớp với src/geometry_analysis.py)
# ---------------------------------------------------------------------------
def edge_canny(gray, low=50, high=180):
    return cv2.Canny(gray, low, high)

def edge_sobel(gray):
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.clip(np.sqrt(sx**2 + sy**2), 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(mag, 50, 255, cv2.THRESH_BINARY)
    return binary

def edge_laplacian(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    lap = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(lap, 15, 255, cv2.THRESH_BINARY)
    return binary

# ---------------------------------------------------------------------------
# Phát hiện đường thẳng (khớp với src/geometry_analysis.py)
# ---------------------------------------------------------------------------
def region_of_interest(img, upper_limit=0.4):
    h, w = img.shape
    mask = np.zeros_like(img)
    start_y = int(h * upper_limit)
    poly = np.array([[(0, h), (w, h), (w, start_y), (0, start_y)]])
    cv2.fillPoly(mask, poly, 255)
    return cv2.bitwise_and(img, mask)

def filter_lines_by_length(lines, min_len=60, max_len=400):
    if lines is None:
        return None
    filtered = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if min_len <= length <= max_len:
            filtered.append(line)
    return filtered if filtered else None

def filter_lines_by_angle(lines, min_angle=20, max_angle=85):
    if lines is None:
        return None
    filtered = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle_deg = math.degrees(math.atan2(abs(y2 - y1), abs(x2 - x1)))
        if min_angle <= angle_deg <= max_angle:
            filtered.append(line)
    return filtered if filtered else None

def draw_probabilistic_hough(edges, img_rgb):
    out = img_rgb.copy()
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=60, minLineLength=80, maxLineGap=80)
    lines = filter_lines_by_length(lines)
    lines = filter_lines_by_angle(lines)
    n = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
            n += 1
    return out, n

def draw_standard_hough(edges, img_rgb):
    out = img_rgb.copy()
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
    n = 0
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
            x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
            cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            n += 1
    return out, n

# ---------------------------------------------------------------------------
# Hàm chuyển figure matplotlib → PIL Image (để Gradio hiển thị)
# ---------------------------------------------------------------------------
def fig_to_pil(fig):
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    pil = Image.frombytes("RGBA", fig.canvas.get_width_height(), buf).convert("RGB")
    plt.close(fig)
    return pil

# ---------------------------------------------------------------------------
# Pipeline handlers
# ---------------------------------------------------------------------------
def run_preprocessing(img_np):
    """Hiển thị 4 bước tiền xử lý."""
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(img_bgr, (400, 400))
    margin = 50
    cropped = resized[margin:400 - margin, margin:400 - margin]
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    denoised = cv2.medianBlur(equalized, ksize=5)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle("Tiền xử lý ảnh", fontsize=14, fontweight="bold")
    for ax, img, title, cmap in zip(
        axes,
        [cropped_rgb, gray, equalized, denoised],
        ["Ảnh gốc (crop)", "Grayscale", "CLAHE (cân bằng sáng)", "Median Blur (giảm nhiễu)"],
        [None, "gray", "gray", "gray"],
    ):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    return fig_to_pil(fig)


def run_edge_comparison(img_np):
    """So sánh 3 phương pháp phát hiện biên."""
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = preprocess_gray(img_bgr)
    original = preprocess_rgb(img_bgr)

    canny = edge_canny(gray)
    sobel = edge_sobel(gray)
    laplacian = edge_laplacian(gray)

    d_canny = np.count_nonzero(canny) / canny.size * 100
    d_sobel = np.count_nonzero(sobel) / sobel.size * 100
    d_lap   = np.count_nonzero(laplacian) / laplacian.size * 100

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle("So sánh phương pháp phát hiện biên", fontsize=14, fontweight="bold")
    for ax, img, title, cmap in zip(
        axes,
        [original, canny, sobel, laplacian],
        ["Ảnh gốc",
         f"Canny\n(density={d_canny:.1f}%)",
         f"Sobel\n(density={d_sobel:.1f}%)",
         f"Laplacian\n(density={d_lap:.1f}%)"],
        [None, "gray", "gray", "gray"],
    ):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    return fig_to_pil(fig)


def run_line_comparison(img_np):
    """So sánh Standard Hough vs Probabilistic Hough."""
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = preprocess_gray(img_bgr)
    original = preprocess_rgb(img_bgr)

    edges = edge_canny(gray)
    roi = region_of_interest(edges)

    prob, n_prob = draw_probabilistic_hough(roi, original)
    std, n_std   = draw_standard_hough(roi, original)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle("So sánh Hough Transform", fontsize=14, fontweight="bold")
    for ax, img, title in zip(
        axes,
        [original, prob, std],
        ["Ảnh gốc",
         f"Probabilistic HoughLinesP\n({n_prob} đoạn, màu đỏ)",
         f"Standard HoughLines\n({n_std} đường, màu xanh lá)"],
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    return fig_to_pil(fig)


def run_object_detection(img_np):
    """Phát hiện đối tượng bằng YOLOv8, hiển thị trước/sau."""
    results = yolo_model(img_np)
    result_bgr = results[0].plot()
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    n_objects = len(results[0].boxes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phát hiện đối tượng — YOLOv8n", fontsize=14, fontweight="bold")
    axes[0].imshow(img_np)
    axes[0].set_title("Ảnh gốc", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(result_rgb)
    axes[1].set_title(f"Kết quả YOLO ({n_objects} đối tượng)", fontsize=12)
    axes[1].axis("off")
    plt.tight_layout()
    return fig_to_pil(fig)

# ---------------------------------------------------------------------------
# Dispatcher chính
# ---------------------------------------------------------------------------
PIPELINE_MAP = {
    "Tiền xử lý (Preprocessing)": run_preprocessing,
    "So sánh phát hiện biên (Canny / Sobel / Laplacian)": run_edge_comparison,
    "So sánh phát hiện đường thẳng (Standard / Probabilistic Hough)": run_line_comparison,
    "Phát hiện đối tượng (YOLOv8)": run_object_detection,
}

def process(img_np, pipeline_name):
    if img_np is None:
        return None
    return PIPELINE_MAP[pipeline_name](img_np)

# ---------------------------------------------------------------------------
# Giao diện Gradio
# ---------------------------------------------------------------------------
with gr.Blocks(title="CV Project 4 Demo") as demo:
    gr.Markdown(
        """
        # CV Project 4 — Demo Tương Tác
        Upload một ảnh (giao thông hoặc cảnh thực tế), chọn pipeline và xem kết quả ngay.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="numpy", label="Ảnh đầu vào")
            pipeline = gr.Dropdown(
                choices=list(PIPELINE_MAP.keys()),
                value="So sánh phát hiện biên (Canny / Sobel / Laplacian)",
                label="Chọn pipeline",
            )
            btn = gr.Button("Chạy", variant="primary")
        with gr.Column(scale=2):
            img_output = gr.Image(label="Kết quả")

    btn.click(fn=process, inputs=[img_input, pipeline], outputs=img_output)

    gr.Examples(
        examples=[
            ["data/raw/img1.jpg", "Phát hiện đối tượng (YOLOv8)"],
            ["data/raw/img2.jpg", "So sánh phát hiện biên (Canny / Sobel / Laplacian)"],
            ["data/raw/img3.jpg", "So sánh phát hiện đường thẳng (Standard / Probabilistic Hough)"],
            ["data/raw/img4.jpg", "Tiền xử lý (Preprocessing)"],
        ],
        inputs=[img_input, pipeline],
    )

if __name__ == "__main__":
    demo.launch(share=False)
