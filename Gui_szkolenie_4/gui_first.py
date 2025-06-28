from datetime import datetime #Potrzebne do automatycznego obliczania wieku
#print(TkVersion) / sprawdzam sobie wersje
import tkinter as tk
from tkinter import ttk
from Data.load_user_data import modify_user_input_for_network
from NeuralNetwork.Network_single_class1 import NNetwork
from Data.One_hot_Encoder import OneHotEncoder
from Data.Transformers import Transformations
import threading
from Data.Decorator_time_logging import log_execution_time

normalization_instance = None
one_hot_instance = None
model_instance = None

#tutaj ładujemy instancje klas stworzone przez Pawła, które będą potrzebne przy załadowaniu formularza
#w momencie uruchomienia okienka GUI, wszystkie instancje klas są jednocześnie tworzone, a nie w momencie klik przycisku zatwierdz
@log_execution_time
def load():
    global normalization_instance, one_hot_instance, model_instance
    normalization_instance = Transformations.load_data()  # instancja klasy normalizującej - ładujemy ją z pliku
    one_hot_instance = OneHotEncoder.load_data()  # plik json jest przypisane do klasy - tworzymy instancje klasy
    model_instance = NNetwork.create_instance()  # Create the model instance
    print(normalization_instance.std_type)
    print("Data loaded successfully!")

# Okienko główne, dajemy root aby obslugiwalo scrollowanie strony
root = tk.Tk()
root.title("Bankowość")
root.geometry("600x600")
root.configure(bg='#f7d6e0')

# Tworzymy canvas + scrollbar
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)

canvas = tk.Canvas(main_frame, bg='#f7d6e0')
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

window = tk.Frame(canvas, bg='#f7d6e0')
canvas.create_window((0, 0), window=window, anchor="nw")

def start_background_loading():
    thread = threading.Thread(target=load, daemon=True)
    thread.start()

start_background_loading()

#ten kodzik ustawia treść okienka na środku
def resize_canvas(event):
    canvas_width = event.width
    canvas.itemconfig(window_id, width=canvas_width)

window_id = canvas.create_window((0, 0), window=window, anchor="nw")
canvas.bind("<Configure>", resize_canvas)



def scroll_canvas(event):
    if event.keysym == 'Down':
        canvas.yview_scroll(1, "units")
    elif event.keysym == 'Up':
        canvas.yview_scroll(-1, "units")

# Bind arrow keys to scroll
root.bind('<Down>', scroll_canvas)
root.bind('<Up>', scroll_canvas)

#ustawiamy kolorki - rozowy :)
style = ttk.Style()
style.configure("TLabel", background="#f7d6e0")
style.configure("TFrame", background="#f7d6e0")
style.configure("TCombobox", fieldbackground="#f7d6e0", background="#f7d6e0")
style.configure("TRadiobutton", background="#f7d6e0")

#konfigurujemy kolor tla suwaka i suwak zeby nie byl bialy - bo ttk nie obsluguje koloru w bg etc.
style.configure("TScale",
    troughcolor="#fcbad3",   # kolor rowka
    background="#f7d6e0",    # kolor tła suwaka
    sliderlength=20
)



#POLA TEKSTOWE

#---IMIĘ I NAZWISKO---
label_name = ttk.Label(window, text="Imię i nazwisko:") #podpisanie pola tekstowego
label_name.pack(pady=10) #pady to odstęp

entry_name = tk.Entry(window) #pole tekstowe do uzupełnienia
entry_name.pack()
#entry samo w sobie przechowuje juz tekst, wiec nie robimy _var = tk.StringVar()

#--PŁEĆ--
label_gender = ttk.Label(window, text ="Płeć:")
label_gender.pack(pady=10)

gender_var = tk.StringVar() #zmienna ktora przechowa wybrana plec po wyborze


#RadioButtony
radio_female = ttk.Radiobutton(window, text="Kobieta", variable=gender_var, value="Female")
radio_female.pack()
radio_male = ttk.Radiobutton(window, text="Mężczyzna", variable=gender_var, value="Male")
radio_male.pack()

gender_var.set("Kobieta")   #domyślnie ustawiam zeby byl odgornie wybor juz na kobiete


#---DATA URODZENIA---
label_birth = ttk.Label(window, text="Data urodzenia:")
label_birth.pack(pady=10) #pady to odstęp
#birth_var = tk.StringVar() #byloby potrzebne, gdyby data byla tekstem, a nie wyborem - do reagowania na zmiany i czytania tekstu w czasie rzeczywistym

#--Data urodzenia--Zrobimy wiek z polem wyboru--
#--Ramka do przechowywanie comboxów--
frame_birth = ttk.Frame(window)
frame_birth.pack(pady=10) #pady=10 to odstęp

#--Dzień--
current_year = datetime.today().year
label_day = ttk.Label(frame_birth, text="Dzień") #napis nad widgetem
label_day.pack(side="left", padx=2) #wywolanie napisu

#days = list(range(1,32)) #lista z wyborem dnia
days = [f"{i:02d}" for i in range(1, 32)] #lista z wyborem dnia --> zamiast 1 wypisuje 01 etc.
day_var = tk.StringVar() #odczyt w czasie rzeczywistym zmienianej wartości

#Widget do wyboru dnia
combo_day = ttk.Combobox(frame_birth, values=days, textvariable=day_var, width=6, state="readonly") #texxtvariable łączy combobox z day_var czyli automatycznym odczytem
combo_day.set("Wybierz")
combo_day.pack(side="left", padx=5)

#textariable - autom. aktualizowanie i pobieranie wartości)
#state - "readonly" - zmusza uzytkownika do wyboru a nie wpisania


#--Miesiąc--analogicznie
label_months = ttk.Label(frame_birth, text="Mies.")
label_months.pack(side="left", padx=2)

#months = list(range(1,13))
months = [f"{i:02d}" for i in range(1, 13)] #lista z wyborem mies. --> zamiast 1 wypisuje 01

month_var = tk.StringVar() #odczyt w czasie rzeczywistym

#Widget do wyboru miesąca
combo_month = ttk.Combobox(frame_birth, values=months, textvariable=month_var, width=6, state="readonly")
combo_month.set("Wybierz") #podpisujemy widget
combo_month.pack(side="left", padx=5)


#---Rok--analogiczne
label_year = ttk.Label(frame_birth, text="Rok")
label_year.pack(side="left", padx=2)

years = list(range(current_year, 1899, -1)) #lista z wiekiem-->pokaże wybór od bieżącego roku do 1900 (-1 oznacza, że ma isc od 2025 w dół)
year_var = tk.StringVar() #w czasie rzeczywistym odczytuje wybraną przez nas wartość

#Widget graficzny do wyboru roku
combo_year = ttk.Combobox(frame_birth, values=years, textvariable=year_var, width=6, state="readonly")
combo_year.set("Wybierz") #podpisujemy widzet jako "rok", coś jak etykieta zanim uzytkownik wybierze rok urodzenia
combo_year.pack(side="left", padx=5)  #dodajemy odstęp


#--Etykieta z wiekiem--
label_age = ttk.Label(window, text="Wiek:")
label_age.pack(pady=10) #pady to odstęp

#--Etykieta ostrzegawcza--wiek < 18
label_waring = ttk.Label(window, text = "", foreground="red")
label_waring.pack()


#--Automatyczne obliczenie wieku oraz walidacja--
age_var = tk.IntVar()
def update_age(*args):
    day = day_var.get()
    month = month_var.get()
    year = year_var.get()

    #sprawdzam czy wszystkie pola mają wybraną wartość:
    if day.isdigit() and month.isdigit() and year.isdigit():
        try:
            #teraz zamiana tekstu w date
            birth_date = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d") #f łączy wartosci w 1 ciag tekstowy
            today = datetime.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            age_var.set(age)

            label_age.config(text=f"Wiek: {age} lat")

            if age < 18:
                label_waring.config(text = "Użytkownik musi mieć co najmniej 18 lat.")
            else:
                label_waring.config(text = "")

        except ValueError:
            label_age.config(text="")
            label_waring.config(text="Nieprawidłowa data")


##--Uruchomienie funkcji poniższej, za kazdym razem gdy zmienie dzien, mies. lub rok

day_var.trace_add("write", update_age) #trace_add("write" - jest wbudowana juz
month_var.trace_add("write", update_age)
year_var.trace_add("write", update_age)



# --Status cywilny--
label_marital = ttk.Label(window, text="Status cywilny:")
label_marital.pack(pady=10)

marital_var = tk.StringVar()
marital_statuses = ["married", "single","divorced"]

combo_marital = ttk.Combobox(window, values=marital_statuses, textvariable=marital_var, state="readonly")
combo_marital.set("Wybierz status")
combo_marital.pack(pady=5)



#--Kraj pochodzenia użytkownika--

label_country = ttk.Label(window, text = "Kraj pochodzenia:")
label_country.pack(pady=10)

#zmienna przechowująca kraje oraz lista krajów
country_var = tk.StringVar()
countries = ["Spain", "Germany", "France"]

combo_country = ttk.Combobox(window, values=countries, textvariable=country_var, state="readonly") #combobox tworzy rozwijaną listę - to widget, readonly- to ze uzytkownik nie moze wpisac, a ma wybrac z listy wartosc
combo_country.set("Wybierz kraj") #set ustawia poczatkowa wartosc zmiennej na tekst Wybierz kraj
combo_country.pack(pady=10)


#---Zawód---
label_zawod = ttk.Label(window, text="Zawód:")
label_zawod.pack(pady=10)

zawod_var = tk.StringVar()
zawod_statuses = ["management", "technician", "entrepreneur", "blue-collar", "unknown", "retired", "admin.", "services", "self-employed", "unemployed", "housemaid", "student"]

combo_zawod = ttk.Combobox(window, values=zawod_statuses, textvariable=zawod_var, state="readonly")
combo_zawod.set("Wybierz zawód")
combo_zawod.pack(pady=5)


# --Poziom wykształcenia--
label_education = ttk.Label(window, text="Poziom wykształcenia:")
label_education.pack(pady=10)

education_var = tk.StringVar()
education_levels = ['tertiary', 'secondary', 'unknown', 'primary']

combo_education = ttk.Combobox(window, values=education_levels, textvariable=education_var, state="readonly")
combo_education.set("Wybierz")
combo_education.pack(pady=10)



#--Ocena Kredytowa--SUWAK--
label_CreditScore = ttk.Label(window, text="Ocena kredytowa:")
label_CreditScore.pack(pady=10)

CreditScore_var = tk.IntVar() #tk.InteVar - zmienna przechowująca wartości l.calkowitych, a tk.StringVar - wartości tekstowe
#--tworzymy suwak do wyboru liczb--
CreditScore_scale = ttk.Scale(window, from_=300, to=850, orient="horizontal", variable=CreditScore_var)  #from_=300 to minimalna wartość jaką może przyjąć suwak [HORIZONTAL - poziomy lub VERTICAL - pionowy]
CreditScore_scale.pack()

#--Funkcja która pokaże wartości na suwaku--
def show_value():
    print(CreditScore_var.get()) #get() odczytuje aktualną wartość

def update_value(*args):
    #aktualizacja etykiety z bieżącą wartością suwaka
    label_value.config(text=f"Wartość oceny kredytowej: {CreditScore_var.get()}")

CreditScore_var.trace_add("write", update_value)

#tworzymy etykiete suwaka
label_value = ttk.Label(window, text=f"Wartość suwaka: {CreditScore_var.get()}")
label_value.pack()

#---Staż w banku---
frame_tenure = ttk.Frame(window)
frame_tenure.pack(pady=15)

label_tenure = ttk.Label(frame_tenure, text="Staż w banku:")
label_tenure.pack(side="left", padx=10)

tenure_var = tk.IntVar()
tenure_options = [i for i in range(0, 31)]  # robimy ranking 0-30 lat

combo_tenure = ttk.Combobox(frame_tenure, values=tenure_options, textvariable=tenure_var, state="readonly", width=10)
combo_tenure.set("Wybierz")
combo_tenure.pack(side="left")


#--Posiada karte kredytowa--

label_hasCard = ttk.Label(window, text ="Posiada kartę kredytową:")
label_hasCard.pack(pady=10)
hasCard_var = tk.StringVar() #zmienna ktora przechowa wybrana odpowiedz po wyborze

#RadioButtony
radio_hasCard = ttk.Radiobutton(window, text="Tak", variable=hasCard_var, value="1")
radio_hasCard.pack()

radio_noCard = ttk.Radiobutton(window, text="Nie", variable=hasCard_var, value="0")
radio_noCard.pack()

hasCard_var.set("Tak")   #domyślnie ustawiam zeby byl odgornie wybor juz na yes


#--Jest aktywnym członkiem--

label_actMember = ttk.Label(window, text ="Aktywny członek:")
label_actMember.pack(pady=10)

actMember_var = tk.StringVar() #zmienna ktora przechowa wybrana plec po wyborze

#RadioButtony
radio_member = ttk.Radiobutton(window, text="Tak", variable=actMember_var, value="1")
radio_member.pack()

radio_noMember = ttk.Radiobutton(window, text="Nie", variable=actMember_var, value="0")
radio_noMember.pack()

actMember_var.set("Tak")   #domyślnie ustawiam zeby byl odgornie wybor juz na tak



# --Czy ma pożyczkę--
label_loan = ttk.Label(window, text="Posiada kredyt:")
label_loan.pack(pady=10)

loan_var = tk.StringVar(value="Nie") #przyjmuje domyślną wartość nie i przechowuje wartość

frame_loan = ttk.Frame(window)
frame_loan.pack()

radio_loan_yes = ttk.Radiobutton(frame_loan, text="Tak", variable=loan_var, value="yes")
radio_loan_yes.pack(side="left", padx=10)

radio_loan_no = ttk.Radiobutton(frame_loan, text="Nie", variable=loan_var, value="no")
radio_loan_no.pack(side="left", padx=10)


#---Saldo---
label_balance = ttk.Label(window, text="Saldo (w zł):")
label_balance.pack(pady=10)

balance_var = tk.StringVar()
entry_balance = tk.Entry(window, textvariable=balance_var)
entry_balance.pack()

#---Szacowane wynagrodzenie---
label_salary = ttk.Label(window, text="Szacowane wynagrodzenie (w zł):")
label_salary.pack(pady=10)

salary_var = tk.StringVar()
entry_salary = tk.Entry(window, textvariable=salary_var)
entry_salary.pack()


#Przyciski i pobieranie danych

# Label to display prediction result
prediction_label = tk.Label(window, text="Chance for user to stay in the bank: ", bg='#f7d6e0', font=('Arial', 10, 'bold'))
prediction_label.pack(pady=10)

def set_label_value(prediction):
    prediction_label.config(text=f"Chance for client to leave the bank: {round(prediction[0], 2)}%")

@log_execution_time
def PobieramDaneFormularza():

    daneFormularza = [
        entry_name.get(),
        CreditScore_var.get(),
        country_var.get(),
        gender_var.get(),
        int(tenure_var.get()),
        int(hasCard_var.get()),
        int(actMember_var.get()),
        float(balance_var.get()),
        age_var.get(),
        zawod_var.get(),
        marital_var.get(),
        education_var.get(),
        float(salary_var.get()),
        loan_var.get()

    ]
    print("input: ")
    print(daneFormularza)
    print()
    return daneFormularza
@log_execution_time
def getprediction():
    pobranedane = PobieramDaneFormularza()
    data_forNN = modify_user_input_for_network(pobranedane, one_hot_instance, normalization_instance)
    print(data_forNN)

    prediction = model_instance.pred(data_forNN[0])
    set_label_value(prediction)



button = ttk.Button(window, command=getprediction, text="Zatwierdź")
button.pack(pady=20)

# DODAJEMY ODSTĘP OD PRZYCISKU ZATWIERDZ
spacer = ttk.Label(window, text="")
spacer.pack(pady=20)

window.mainloop() #odswiezanie okienka, uruchomienie pętli aplikacji


### backend możliwość wysyłąnia requestu i wysyłany odpowiedź 