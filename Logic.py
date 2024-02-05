import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Variables to store training results
error_norm_by_epoch = []
weigth_evolution = []
initial_weights = None
final_weights = None
num_of_epochs = 0
allowable_error = 0

def train_perceptron(eta, epochs, file_path, progress_bar):
    global error_norm_by_epoch, weigth_evolution, initial_weights, num_of_epochs, final_weights

    error_norm_by_epoch.clear()
    weigth_evolution.clear()

    delimitador = ';'
    
    data_frame = pd.read_csv(
        file_path, 
        delimiter=delimitador,
        header=None
        )

    num_characteristics = len(data_frame.columns) - 1

    weigths = np.random.uniform(low=0, high=1, size=(num_characteristics + 1, 1)).round(4)
    columns_x = np.hstack([data_frame.iloc[:, :-1].values, np.ones((data_frame.shape[0], 1))])
    columns_y = np.array(data_frame.iloc[:, -1])

    initial_weights = weigths.copy()
    num_of_epochs = epochs
    print("Initial Weigths:")
    print(weigths)

    for i in range(num_characteristics + 1):
        weigth_evolution.append([])

    for epoch in range(epochs):
        u = np.dot(columns_x, weigths)
        calculated_y = np.where(u >= 0, 1, 0).reshape(-1, 1)
        found_errors = columns_y.reshape(-1, 1) - calculated_y

        norma_error = np.linalg.norm(found_errors)
        error_norm_by_epoch.append(norma_error)

        for i in range(num_characteristics + 1):
            weigth_evolution[i].append(weigths[i, 0])

        product_found_errors = np.dot(columns_x.T, found_errors)
        delta_w = eta * product_found_errors
        weigths += np.round(delta_w, 4)

        # Update the progress bar
        progress_bar['value'] = (epoch + 1) / epochs * 100
        progress_bar.update()
    final_weights = weigths

def visualize_results():
    global error_norm_by_epoch, weigth_evolution
    # width, height
    plt.figure(figsize=(5, 7))
    # 2 rows, 1 column, 1 graph
    plt.subplot(2, 1, 1)
    plt.plot(range(1, len(error_norm_by_epoch) + 1), error_norm_by_epoch)
    plt.title('EVOLUCIÓN DE LA NORMA DEL ERROR (|e|)')
    plt.xlabel('Épocas')
    plt.ylabel('Norma del Error')

    # 2 rows, 1 column, 1 graph
    plt.subplot(2, 1, 2)
    for i, weigths_epoca in enumerate(weigth_evolution):
        plt.plot(range(1, len(weigths_epoca) + 1), weigths_epoca, label=f'Peso {i + 1}')
    plt.title('EVOLUCIÓN DEL VALOR DE LOS PESOS (W)')
    plt.xlabel('Épocas')
    plt.ylabel('Valor del Peso')
    plt.legend()

    # Disable scientific notation on the y-axis
    plt.ticklabel_format(style='plain', axis='y')

    # Get the manager of the current figure
    fig_manager = plt.get_current_fig_manager()

    # Get screen dimensions
    screen_width = fig_manager.window.winfo_screenwidth()
    screen_height = fig_manager.window.winfo_screenheight()

    # Get window dimensions
    window_width = 550 
    window_height = 750  

    # Calculate the coordinates to center the window
    x_coordinate = (screen_width - window_width) // 2
    y_coordinate = (screen_height - window_height - 50) // 2

    # Set window coordinates
    fig_manager.window.geometry(f'{window_width}x{window_height}+{x_coordinate}+{y_coordinate}')

    plt.tight_layout()
    plt.show()

def get_weigths():
    return initial_weights, final_weights, num_of_epochs, allowable_error
