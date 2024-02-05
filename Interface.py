from tkinter import filedialog, ttk
import Logic as pl
import tkinter as tk
import numpy as np
import threading
import os

error_fields = []
training = False

def error_dialog(eta, epochs, archivo):
    if not len(archivo):
        if "Seleccionar archivo [CSV]" not in error_fields:
            error_fields.append("Seleccionar archivo [CSV]")
    else:
        if "Seleccionar archivo [CSV]" in error_fields:
            error_fields.remove("Seleccionar archivo [CSV]")

    if not len(eta):
        if "Tasa de Aprendizaje (ETA)" not in error_fields:
            error_fields.append("Tasa de Aprendizaje (ETA)")
    else:
        if "Tasa de Aprendizaje (ETA)" in error_fields:
            error_fields.remove("Tasa de Aprendizaje (ETA)")

    if not len(epochs):
        if "Épocas" not in error_fields:
            error_fields.append("Épocas")
    else:
        if "Épocas" in error_fields:
            error_fields.remove("Épocas")

    if error_fields:
        error_message = f"Los siguientes campos son requeridos y están vacíos: {', '.join(error_fields)}"
        tk.messagebox.showinfo("Campos Vacíos", error_message)
        return False
    else:
        return True

def select_file():
    initial_dir = "C:/Users/1108361138/OneDrive/Escritorio/Carpetas del Desktop/UP/8VO CUATRIMESTRE/INTELIGENCIA ARTIFICIAL/IA_C2_A2/213469_PERCEPTRON/"
    
    if not os.path.exists(initial_dir):
        initial_dir = "/"
    
    filename = filedialog.askopenfilename(
        initialdir=initial_dir, 
        title="Select file",
        filetypes=(("csv files", "*.csv"), 
        ("all files", "*.*"))
        )
    message = f"Archivo extraído desde: {filename}"
    tk.messagebox.showinfo("CSV Cargado Éxitosamente", message)
    file_label.config(text=filename)
    file_label_message.config(text=f"Archivo CSV Cargado de Forma Correcta")
    return filename

def start_training():
    global training
    result = error_dialog(eta_entry.get(), epochs_entry.get(), file_label.cget("text"))
    if result:
        file_path = file_label.cget("text")
        if file_path:
            eta = float(eta_entry.get())
            epochs = int(epochs_entry.get())
            threading.Thread(target=lambda: pl.train_perceptron(eta, epochs, file_path, progress_bar)).start()
            training = True


def visualize_graphs():
    if not training:
        error_message = f"Para generar una gráfica, debe haber un entrenamiento previo"
        tk.messagebox.showinfo("Gráficas", error_message)
    else:
        pl.visualize_results()

def generate_report():
    if not training:
        error_message = f"Para generar un reporte, debe haber un entrenamiento previo"
        tk.messagebox.showinfo("Reportes", error_message)
    else:
        np.set_printoptions(precision=4, suppress=True)
        pesos_iniciales, pesos_finales, epochs, error = pl.get_weigths()
        reporte = (
            f"Tasa de aprendizaje (ETA): {eta_entry.get()}\n"
            f"Número de Épocas: {epochs}\n"
            f"Error permisible: {error}\n\n"
            f"Configuracion de pesos iniciales:\n{pesos_iniciales}\n\n"
            f"Configuracion de pesos finales:\n{pesos_finales}"
            )
        tk.messagebox.showinfo("Reporte General", reporte)

root = tk.Tk()
root.title("Neurona Artificial: Perceptrón")
root.geometry('600x400')

style = ttk.Style()
style.theme_use('clam')

# Principal container
main_frame = ttk.Frame(root, padding="10 10 10 10")
main_frame.pack(fill=tk.BOTH, expand=True)

# Subcontainer for elements
frame = ttk.Frame(main_frame)
frame.pack(fill=tk.BOTH, expand=True)

# New container for buttons
buttons_container = ttk.Frame(main_frame)
buttons_container.pack(pady=5, fill=tk.X)

# Progress Bar
progress_bar = ttk.Progressbar(main_frame, orient="horizontal", length=600, mode="determinate")
progress_bar.pack(pady=0, fill=tk.X)

ttk.Label(frame, text="Tasa de Aprendizaje (ETA):").pack(pady=5)
eta_entry = ttk.Entry(frame)
eta_entry.pack()

ttk.Label(frame, text="Épocas:").pack(pady=5)
epochs_entry = ttk.Entry(frame)
epochs_entry.pack()

ttk.Button(frame, text="Seleccionar archivo [CSV]", command=select_file).pack(pady=10)
file_label = ttk.Label(frame, text="")
file_label.pack(pady=5)
file_label_message = ttk.Label(frame, text="")
file_label_message.pack(pady=5)

# Buttons in the new container
visualize_graphs_button = ttk.Button(buttons_container, text="Mostrar Gráficas", command=visualize_graphs)
visualize_graphs_button.pack(side=tk.LEFT, padx=5)
visualize_graphs_button['state'] = 'normal'

ttk.Button(buttons_container, text="Generar Reporte", command=generate_report).pack(side=tk.LEFT, padx=5)
ttk.Button(buttons_container, text="Comenzar Entrenamiento", command=start_training).pack(side=tk.RIGHT, padx=5)

# Get screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Get window dimensions
window_width = 420
window_height = 320

# Calculate the coordinates to center the window
x_coordinate = (screen_width - window_width) // 2
y_coordinate = (screen_height - window_height) // 2

# Set window coordinates
root.geometry(f'{window_width}x{window_height}+{x_coordinate}+{y_coordinate}')

# Run the main loop
root.mainloop()