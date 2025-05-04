import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import cv2
import numpy as np
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import transforms
from xgboost import XGBClassifier

class FeedForwardMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FeedForwardMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

def preprocess_image(img):
    img_resized = cv2.resize(img, (256, 256))
    img_norm = img_resized / 255.0
    img_denoised = cv2.GaussianBlur(img_norm, (3, 3), 0)
    img_denoised = (img_denoised * 255).astype(np.uint8)
    lab = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    gray_scaled = cv2.cvtColor(img_clahe, cv2.COLOR_BGR2GRAY)
    sharpen_kernel = np.array([[0, -0.5, 0],
                               [-0.5, 3, -0.5],
                               [0, -0.5, 0]], dtype=np.float32)
    blurred = cv2.GaussianBlur(gray_scaled, (3, 3), 0)
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)
    kernel = np.ones((3, 3), np.uint8)
    morph_clean = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
    return img_clahe, morph_clean

# ==== GUI Class ====
class ObjectDetectionGUI:
    def __init__(self, root):
        self.feature_extractor = mobilenet_v2(pretrained=True).features
        self.feature_extractor.eval()
        self.model = FeedForwardMLP(1280, 5)
        state_dict = torch.load("D:/Downloads/FeedForwardMLP_epoch_14.pth", map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.xgb_model = XGBClassifier()
        self.xgb_model.load_model("D:/Downloads/xgb_classifier_model.json")

        self.class_names = ['Rose', 'Tulip', 'Daisy', 'Sunflower', 'Dandelion']

        self.root = root
        self.root.title("Object Detection System")
        self.root.geometry("800x600")

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.instructions = tk.Label(self.main_frame, text="Upload an image and click 'Detect Objects' to start.", font=('Arial', 12))
        self.instructions.pack(pady=5)

        # Model selection
        self.model_choice = tk.StringVar(value="MLP")
        self.model_selector_frame = tk.Frame(self.main_frame)
        self.model_selector_frame.pack(pady=5)
        tk.Label(self.model_selector_frame, text="Choose model:").pack(side=tk.LEFT)
        tk.Radiobutton(self.model_selector_frame, text="MLP", variable=self.model_choice, value="MLP").pack(side=tk.LEFT)
        tk.Radiobutton(self.model_selector_frame, text="XGBoost", variable=self.model_choice, value="XGB").pack(side=tk.LEFT)

        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(pady=10)

        self.upload_button = tk.Button(self.button_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.grid(row=0, column=0, padx=5)

        self.detect_button = tk.Button(self.button_frame, text="Detect Objects", command=self.detect_objects)
        self.detect_button.grid(row=0, column=1, padx=5)

        self.reset_button = tk.Button(self.button_frame, text="Reset", command=self.reset)
        self.reset_button.grid(row=0, column=2, padx=5)

        self.canvas = tk.Canvas(self.main_frame, width=600, height=400, bg='lightgray')
        self.canvas.pack(pady=10)

        self.result_label = tk.Label(self.main_frame, text="Detection Results:", justify=tk.LEFT, anchor='nw')
        self.result_label.pack(pady=5, fill=tk.X)

        self.status = tk.Label(root, text="Welcome to the Object Detection System", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.image_path = file_path
            self.original_image = Image.open(file_path)
            self.cv_image = cv2.imread(file_path)
            self.cv_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
            self.display_image(file_path)
            self.status.config(text="Image uploaded successfully.")

    def display_image(self, file_path):
        image = Image.open(file_path)
        image = image.resize((600, 400), Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def reset(self):
        self.canvas.delete("all")
        self.result_label.config(text="Detection Results:")
        self.status.config(text="Reset complete. Ready for a new image.")

    def detect_objects(self):
        if not hasattr(self, 'cv_image'):
            messagebox.showwarning("Warning", "Please upload an image first.")
            return

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        input_image = transform(self.cv_image).unsqueeze(0)

        with torch.no_grad():
            features = self.feature_extractor(input_image)
            features = torch.nn.functional.adaptive_avg_pool2d(features, 1).view(1, -1)

            if self.model_choice.get() == "MLP":
                outputs = self.model(features)
                probs = torch.nn.functional.softmax(outputs, dim=1)
            else:  # XGBoost
                np_features = features.numpy()
                probs_np = self.xgb_model.predict_proba(np_features)
                probs = torch.tensor(probs_np)

            score, pred = torch.max(probs, 1)
            score = score.item()
            class_idx = pred.item()

            if score >= 0.8:
                label = self.class_names[class_idx]
                x1, y1, x2, y2 = 100, 100, 300, 300  # Dummy box
                results = [(x1, y1, x2, y2, label, score, class_idx)]
                self._draw_results(results)
            else:
                self.result_label.config(
                    text=f"Detection Results:\nNo confident prediction (score < 0.80).\n"
                         f"Predicted Class Index: {class_idx}, Score: {score:.2f}"
                )
                self.status.config(text="Detection complete (no high-confidence class).")

    def _draw_results(self, results):
        self.canvas.delete("all")
        self.display_image(self.image_path)
        results_text = "Detection Results:\n"

        for idx, (x1, y1, x2, y2, label, score, class_idx) in enumerate(results, start=1):
            self.canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)
            text_id = self.canvas.create_text(x1, y1, anchor=tk.NW, text=label, fill='white', font=('Arial', 12, 'bold'))
            bbox = self.canvas.bbox(text_id)
            self.canvas.create_rectangle(bbox, fill='black', outline='black')
            self.canvas.tag_raise(text_id)
            results_text += f"{idx}. {label} - Score: {score:.2f}\nBounding Box: ({x1}, {y1}), ({x2}, {y2})\n"

        self.result_label.config(text=results_text)
        self.status.config(text="Detection complete.")

# ==== Main Entry Point ====
if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionGUI(root)
    root.mainloop()