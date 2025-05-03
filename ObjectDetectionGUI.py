import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk





class ObjectDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection System")
        self.root.geometry("800x600")

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.instructions = tk.Label(self.main_frame, text="Upload an image and click 'Detect Objects' to start.", font=('Arial', 12))
        self.instructions.pack(pady=5)

        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(pady=10)

        self.upload_button = tk.Button(self.button_frame, text="Upload Image", command=self.upload_image)
        self.upload_button.grid(row=0, column=0, padx=5)

        # Example detection results --- remove this when you have the real results from the model 
        # and pass the results to the detect_objects function in the detect_button command
        # to get image write self.
        example_results = [
            (50, 50, 150, 150, 'Object1', 0.95),
            (200, 200, 300, 300, 'Object2', 0.93)
                          ]

        self.detect_button = tk.Button(self.button_frame, text="Detect Objects", command=lambda: self.detect_objects(example_results))
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
            self.original_image = Image.open(file_path)
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

    def detect_objects(self, results):
        if not hasattr(self, 'original_image'):
            messagebox.showwarning("Warning", "Please upload an image first.")
            return

        
        results_text = "Detection Results:\n"

        # Draw bounding boxes and labels
        for idx, (x1, y1, x2, y2, label, score) in enumerate(results, start=1):
                self.canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)
                # Draw text with background
                text_id = self.canvas.create_text(x1, y1, anchor=tk.NW, text=label, fill='white', font=('Arial', 12, 'bold'))
                bbox = self.canvas.bbox(text_id)
                self.canvas.create_rectangle(bbox, fill='black', outline='black')
                self.canvas.tag_raise(text_id)  # Ensure text is above the rectangle
                results_text += (f"{idx}. {label} - Score: {score:.2f}\n"
                                 f"   Bounding Box: ({x1:.0f}, {y1:.0f}), ({x2:.0f}, {y2:.0f})\n")

        # Update the result label
        self.result_label.config(text=results_text)
        self.status.config(text="Object detection completed.")

        messagebox.showinfo("Info", "Object detection completed.")




if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionGUI(root)
    root.mainloop()
 