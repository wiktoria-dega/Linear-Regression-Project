import tkinter as tk
from tkinter import ttk
import save_read_model_scaler
import numpy as np
from PIL import Image, ImageTk


loaded_model, loaded_scaler = save_read_model_scaler.load_model_scaler()


def create_widgets(root):
    global sqft_living, sqft_lot, bathrooms, floors, waterfront, view, condition, \
    cond_scale, grade, sqft_basement, sqft_living15, sqft_lot15, dist_bellevue, \
    dist_ns, nmb_rooms, age

    
    welcome_label = ttk.Label(root, text=(
    "Value your property by entering the metric values below!"),
        font=('Helvetica', 16), wraplength=750, justify="center")
    welcome_label.grid(row=0, column=0, columnspan=2, pady=(20, 40))

    
    sqft_living = ttk.Entry(root)
    sqft_living.grid(row=2, column=1, padx=10, pady=5)
    sqft_lot = ttk.Entry(root)
    sqft_lot.grid(row=3, column=1, padx=10, pady=5)
    bathrooms = ttk.Entry(root)
    bathrooms.grid(row=4, column=1, padx=10, pady=5)
    floors = ttk.Entry(root)
    floors.grid(row=5, column=1, padx=10, pady=5)
    waterfront = ttk.Entry(root)
    waterfront.grid(row=6, column=1, padx=10, pady=5)
    view = ttk.Entry(root)
    view.grid(row=7, column=1, padx=10, pady=5)
    condition = ttk.Entry(root)
    condition.grid(row=8, column=1, padx=10, pady=5)
    cond_scale = ttk.Entry(root)
    cond_scale.grid(row=9, column=1, padx=10, pady=5)
    grade = ttk.Entry(root)
    grade.grid(row=10, column=1, padx=10, pady=5)
    sqft_basement = ttk.Entry(root)
    sqft_basement.grid(row=11, column=1, padx=10, pady=5)
    sqft_living15 = ttk.Entry(root)
    sqft_living15.grid(row=12, column=1, padx=10, pady=5)
    sqft_lot15 = ttk.Entry(root)
    sqft_lot15.grid(row=13, column=1, padx=10, pady=5)
    dist_bellevue = ttk.Entry(root)
    dist_bellevue.grid(row=14, column=1, padx=10, pady=5)
    dist_ns = ttk.Entry(root)
    dist_ns.grid(row=15, column=1, padx=10, pady=5)
    nmb_rooms = ttk.Entry(root)
    nmb_rooms.grid(row=16, column=1, padx=10, pady=5)
    age = ttk.Entry(root)
    age.grid(row=17, column=1, padx=10, pady=5)

    
    labels = ['Living area (sq ft)', 'Plot area (sq ft)', 'Number of bathrooms',
              'Number of floors', 'Water view (0 - No, 1 - Yes)',
              'Assessment of the view (0-4)',
              'House condition assessment (1-5)',
              'Scale of house condition (0 - No, 1 - Yes)',
              'Evaluation of the quality of construction and design of the house (1-13)',
              'Basement area (sq ft, 0 if none)', 'Living area of the nearest 15 neighbors (sq ft)',
              'Plot area of 15 nearest neighbors (sq ft)', 'Distance to Bellevue city (km)',
              'Distance to Northwest Seattle (km)', 'Number of rooms (bedrooms and bathrooms)',
              'Age of house']
    for i, label_text in enumerate(labels):
        label = ttk.Label(root, text=label_text)
        label.grid(row=i+2, column=0, sticky='w', padx=10, pady=5)

    
    predict_button = ttk.Button(root, text="Predict", command=predict_disease)
    predict_button.grid(row=12, column=0, columnspan=2, padx=5, pady=5)

    
    global result_label
    result_label = ttk.Label(root, text="Prediction: None")
    result_label.grid(row=13, column=0, columnspan=2, padx=5, pady=5)
    
    '''
    image = Image.open('heart.png')
    photo = ImageTk.PhotoImage(image)
    image_label = tk.Label(root, image=photo)
    image_label.image = photo  
    image_label.grid(row=14, column=0, columnspan=2, sticky='ew')
    '''


def predict_disease():
    try:
        # Przygotowanie danych wejściowych
        inputs = np.array([[float(sqft_living.get()), float(sqft_lot.get()), float(bathrooms.get()),
                            float(floors.get()), int(waterfront.get()), int(view.get()),
                            int(condition.get()), int(cond_scale.get()), int(grade.get()),
                            float(sqft_basement.get()), float(sqft_living15.get()),
                            float(sqft_lot15.get()), float(dist_bellevue.get()),
                            float(dist_ns.get()), float(nmb_rooms.get()), int(age.get())]])
        inputs_scaled = loaded_scaler.transform(inputs)
        
        # Przewidywanie
        prediction = loaded_model.predict(inputs_scaled)
        
        prediction_exp = np.exp(prediction)
    
        result_label.config(text=(f"Prediction of house price: ${prediction_exp[0]:,.2f}") )
    except Exception as e:
        result_label.config(text=f'Error: {str(e)}')

def main():
    root = tk.Tk()
    root.title("House Price Prediction")
    root.geometry('760x1000')
    create_widgets(root)
    root.mainloop()


if __name__ == "__main__":
    main()
    