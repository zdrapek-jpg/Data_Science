import tkinter as tk
from tkinter import ttk, RIGHT, Y, VERTICAL
from Data.load_user_data import modify_user_input_for_network
from NeuralNetwork.Network_single_class import NNetwork
from Data.One_hot_Encoder import OneHotEncoder
from Data.Transformers import Transformations
import threading

normalization_instance = None
one_hot_instance = None
model_instance = None


def load():
    global normalization_instance, one_hot_instance, model_instance
    normalization_instance = Transformations.load_data()  # Load the transformation data
    one_hot_instance = OneHotEncoder.load_data()  # Load one-hot encoding data
    model_instance = NNetwork.create_instance()  # Create the model instance
    print("Data loaded successfully!")


def app():
    # Okienko główne
    window = tk.Tk()
    window.title("Bankowość")
    window.geometry("700x700")
    window.configure(bg='#f7d6e0')  # Kolor okienka - różowy ;)

    # Tworzymy ramkę, aby móc przewijać zawartość
    frame = tk.Frame(window, bg='#f7d6e0')
    frame.pack(fill=tk.BOTH, expand=True)

    # Tworzymy Canvas do umieszczenia zawartości
    canvas = tk.Canvas(frame, bg='#f7d6e0')
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Dodajemy Scrollbar
    scroll_bar = tk.Scrollbar(frame, orient=VERTICAL, command=canvas.yview)
    scroll_bar.pack(side=RIGHT, fill=Y)

    # Konfigurujemy Canvas aby współpracował ze Scrollbarem
    canvas.configure(yscrollcommand=scroll_bar.set)
    canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

    # Tworzymy nową ramkę wewnątrz Canvas
    form_frame = tk.Frame(canvas, bg='#f7d6e0')
    canvas.create_window((0, 0), window=form_frame, anchor='nw')
    # Definicje pól
    fields = [
        ('Surname', 'String dowolne'),
        ('Country', 'Spain, Germany, France'),
        ('Gender', 'Male lub Female'),
        ('EstimatedSalary', 'int lub float, 2 miejsca po przecinku'),
        ('Age', '22 - 61'),
        ('Job', 'management, technician, entrepreneur, ...'),
        ('Marital', 'married, single, divorced'),
        ('Education', 'tertiary, secondary, unknown, primary')
    ]

    entries = {}

    # Tworzenie siatki pól
    for idx, (label, placeholder) in enumerate(fields):
        tk.Label(form_frame, text=f"{label}:", bg='#f7d6e0', font=('Arial', 10, 'bold')).grid(row=idx, column=0,
                                                                                              padx=10, pady=5,
                                                                                              sticky='w')
        if label == 'Marital':
            options = ['married', 'single', 'divorced']
            entry = ttk.Combobox(form_frame, values=options, width=37)
            entry.current(0)  # Set default to 'married'
        elif label == 'Education':
            options = ['tertiary', 'secondary', 'unknown', 'primary']
            entry = ttk.Combobox(form_frame, values=options, width=37)
            entry.current(0)  # Set default to 'tertiary'
        elif label == 'Gender':
            options = ['Male', 'Female']
            entry = ttk.Combobox(form_frame, values=options, width=37)
            entry.current(0)  # Set default to 'Male'
        elif label == 'Country':
            options = ['Spain', 'Germany', 'France']
            entry = ttk.Combobox(form_frame, values=options, width=37)
            entry.current(0)  # Set default to 'Spain'
        elif label == 'Job':
            options = [
                "management", "technician", "entrepreneur", "blue-collar", "unknown",
                "retired", "admin.", "services", "self-employed", "unemployed",
                "housemaid", "student"
            ]
            entry = ttk.Combobox(form_frame, values=options, width=37)
            entry.current(0)  # Set default to
        else:
            entry = tk.Entry(form_frame, width=40)
        entry.grid(row=idx, column=1, padx=10, pady=5)
        entry.insert(0, placeholder)  # Default value inserted into the field
        entries[label] = entry
    # Function to update progress bars
    def update_label(scale, entry):
        value = int(float(scale.get()))
        entry.delete(0, tk.END)
        entry.insert(0, str(value))

    # CreditScore
    tk.Label(form_frame, text="CreditScore:", bg='#f7d6e0', font=('Arial', 10, 'bold')).grid(row=8, column=0, padx=10, pady=5, sticky='w')
    creditscore_value_label = tk.Entry(form_frame, width=10)
    creditscore_value_label.grid(row=8, column=2, padx=10, pady=5)
    creditscore_scale = ttk.Scale(form_frame, from_=0, to=1000, orient='horizontal', command=lambda value: update_label(creditscore_scale, creditscore_value_label))
    creditscore_scale.grid(row=8, column=1, padx=10, pady=5)
    entries['CreditScore'] = creditscore_value_label

    # Tenure
    tk.Label(form_frame, text="Tenure:", bg='#f7d6e0', font=('Arial', 10, 'bold')).grid(row=9, column=0, padx=10, pady=5, sticky='w')
    tenure_value_label = tk.Entry(form_frame, width=10)
    tenure_value_label.grid(row=9, column=2, padx=10, pady=5)
    tenure_scale = ttk.Scale(form_frame, from_=0, to=10, orient='horizontal', command=lambda value: update_label(tenure_scale, tenure_value_label))
    tenure_scale.grid(row=9, column=1, padx=10, pady=5)
    entries['Tenure'] = tenure_value_label

    # Balance
    tk.Label(form_frame, text="Balance:", bg='#f7d6e0', font=('Arial', 10, 'bold')).grid(row=10, column=0, padx=10, pady=5, sticky='w')
    balance_value_label = tk.Entry(form_frame, width=10)
    balance_value_label.grid(row=10, column=2, padx=10, pady=5)
    balance_scale = ttk.Scale(form_frame, from_=-100000, to=100000, orient='horizontal', command=lambda value: update_label(balance_scale, balance_value_label))
    balance_scale.grid(row=10, column=1, padx=10, pady=5)
    entries['Balance'] = balance_value_label

    # HasCrCard
    tk.Label(form_frame, text='HasCrCard:', bg='#f7d6e0', font=('Arial', 10, 'bold')).grid(row=11, column=0, padx=10,
                                                                                           pady=5, sticky='w')
    hascrcard_var = tk.IntVar()
    tk.Radiobutton(form_frame, text='Yes', variable=hascrcard_var, value=1, bg='#f7d6e0').grid(row=11, column=1,
                                                                                               sticky='w')
    tk.Radiobutton(form_frame, text='No', variable=hascrcard_var, value=0, bg='#f7d6e0').grid(row=11, column=1, padx=60)
    # IsActiveMember
    tk.Label(form_frame, text='IsActiveMember:', bg='#f7d6e0', font=('Arial', 10, 'bold')).grid(row=12, column=0,
                                                                                                padx=10, pady=5,
                                                                                                sticky='w')
    isactive_var = tk.IntVar()
    tk.Radiobutton(form_frame, text='Yes', variable=isactive_var, value=1, bg='#f7d6e0').grid(row=12, column=1,                                                                      sticky='w')
    tk.Radiobutton(form_frame, text='No', variable=isactive_var, value=0, bg='#f7d6e0').grid(row=12, column=1, padx=60)
    # Loan
    tk.Label(form_frame, text='Loan:', bg='#f7d6e0', font=('Arial', 10, 'bold')).grid(row=13, column=0, padx=10, pady=5,
                                                                                      sticky='w')
    loan_var = tk.IntVar()
    tk.Radiobutton(form_frame, text='Yes', variable=loan_var, value=1, bg='#f7d6e0').grid(row=13, column=1, sticky='w')
    tk.Radiobutton(form_frame, text='No', variable=loan_var, value=0, bg='#f7d6e0').grid(row=13, column=1, padx=60)
    def collect_data():
        user_data = [
            entries['Surname'].get(),
            int(creditscore_value_label.get()),
            entries['Country'].get(),
            entries['Gender'].get(),
            int(tenure_value_label.get()),
            hascrcard_var.get(),
            isactive_var.get(),
            float(balance_value_label.get()),
            int(entries['Age'].get()),
            entries['Job'].get(),
            entries['Marital'].get(),
            entries['Education'].get(),
            float(entries['EstimatedSalary'].get()),
            'yes' if loan_var.get() == 1 else 'no'
        ]
        print(user_data)
        return user_data

    # Function to handle the submit button click (prediction)
    def on_submit_button():
        # Disable the submit button to avoid multiple clicks
        submit_button.config(state=tk.DISABLED)
        # Start prediction task in a separate thread
        threading.Thread(target=make_prediction).start()

    def make_prediction():
        user_data = collect_data()
        data_forNN = modify_user_input_for_network(user_data, one_hot_instance, normalization_instance)

        prediction = model_instance.pred(data_forNN[0])

        # Update the UI with the prediction using the after() method
        window.after(0, update_prediction_label, prediction[0])
        # Re-enable the submit button after prediction
        window.after(0, enable_submit_button)


    def update_prediction_label(prediction):
        prediction_label.config(text=f"Chance for client to leave the bank: {round(prediction, 2)}%")

    # Function to enable the submit button after the prediction is done
    def enable_submit_button():
        submit_button.config(state=tk.NORMAL)
    # Submit Button
    submit_button = tk.Button(form_frame, text="Submit", command=on_submit_button, bg="#d6a4a4", font=("Arial", 10, "bold"))
    submit_button.grid(row=14, column=1, columnspan=2, pady=10)

    # Label to display prediction result
    prediction_label = tk.Label(form_frame, text="Chance for user to stay in the bank: ", bg='#f7d6e0', font=('Arial', 10, 'bold'))
    prediction_label.grid(row=14, column=0, sticky='w')

    # Load data when the application starts in a separate thread
    threading.Thread(target=load).start()

    window.mainloop()


app()

