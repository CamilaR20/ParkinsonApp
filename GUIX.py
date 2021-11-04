from tkinter import *
import tkinter as tk
import firebase_admin
import cv2
import imutils
import mediapipe as mp
import csv
import os
from firebase_admin import credentials
from firebase_admin import storage
from tkinter import ttk
from zipfile import ZipFile
from tkinter import filedialog
from tkinter import Label
from Parkinson_movements import *
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.widgets import Cursor, Button
from cursorClass import *
from PIL import Image
from PIL import ImageTk



cred = credentials.Certificate('/Users/camilaroa/PycharmProjects/parkinsonApp/serviceAccountKey.json')
firebase_admin.initialize_app(cred, {'storageBucket': 'parkinsondata.appspot.com'})

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./assetsp")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


def search_vid():
    global dates_list
    global bucket
    prefix = patient_id.get()
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix=str(prefix)+'/')
    dates = []
    fecha.set("Seleccione fecha")
    movimiento.set("Seleccione movimiento")
    mano.set("Seleccione mano")
    for blob in blobs:
        carpeta = blob.name
        result = carpeta.find('/')
        date = carpeta[result+1:result+14]
        dates.append(date)
    date_entry['menu'].delete(0, 'end')
    for choice in dates:
        date_entry['menu'].add_command(label=choice, command=tk._setit(fecha, choice))


def carpeta_descarga():
    global download_folder
    global folder_id
    download_folder = filedialog.askdirectory()
    canvas.delete(folder_id)
    folder_id = canvas.create_text(60.0, 127.0, anchor="nw", width=270, text=download_folder, fill="#000000", font=("Roboto", 14 * -1))


def visualizar():
    global cap
    global img
    global posicion
    fps = cap.get(cv2.CAP_PROP_FPS)
    iniciar_btn['state'] = tk.DISABLED
    ret, frame = cap.read()
    if ret:
        frame = imutils.resize(frame, height=600)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)

        video_label.configure(image=img)
        video_label.image = img
        video_label.after(int((1/fps)*1000), visualizar)
    else:
        show_first_frame()
        iniciar_btn['state'] = tk.NORMAL


def btn_iniciar():
    global cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, posicion)
    visualizar()


def btn_pausar():
    global img
    global cap
    global posicion
    posicion = cap.get(cv2.CAP_PROP_POS_FRAMES)
    cap.release()
    video_label.image = img
    # ret, current_frame = cap.read(posicion)
    # current_frame = imutils.resize(current_frame, height=600)
    # current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    #
    # im = Image.fromarray(current_frame)
    # img = ImageTk.PhotoImage(image=im)
    #
    # video_label.configure(image=img)
    # video_label.image = img


def show_first_frame():
    global cap
    cap = cv2.VideoCapture(video_name)
    ret, frame = cap.read()
    frame = imutils.resize(frame, height=600)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    im = Image.fromarray(frame)
    img = ImageTk.PhotoImage(image=im)

    video_label.configure(image=img)
    video_label.image = img


# defining a function that will get the patient id and print them on the screen
def adquirir_video():
    global id
    global mov
    global video_name
    global video_chosen
    id = patient_id.get()
    mov = movimiento.get()
    dat = fecha.get()
    # blob = bucket.blob(str(id) + '/' + str(dat) + '/test.zip')
    # blob.download_to_filename(download_folder + "/intento1.zip")
    # with ZipFile(download_folder + "/intento1.zip", 'r') as zip:
    #     zip.printdir()
    #     zip.extractall(download_folder + '/diagnostico_parkinson')

    if mano.get() == 'Derecha':
        if mov == 'Golpeteo de dedos':
            mov_eval = 'fingertap_r'
        elif mov == 'Prono supinacion':
            mov_eval = 'pronosup_r'
        elif mov == 'Cierre de puño':
            mov_eval = 'fist_r'
    elif mano.get() == 'Izquierda':
        if mov == 'Golpeteo de dedos':
            mov_eval = 'fingertap_l'
        elif mov == 'Prono supinacion':
            mov_eval = 'pronosup_l'
        elif mov == 'Cierre de puño':
            mov_eval = 'fist_l'

    #video_name = download_folder + '/temp/' + mov_eval + '.mp4'
    video_name = '/Users/camilaroa/Desktop/mytest/1234/26-09-2021, 16:55, OFF/fingertap_l.mp4'
    show_first_frame()
    archivo_csv(video_name, download_folder, mov_eval, dat)


def archivo_csv(name_video, folder_download, mov_eval, dat):
    global nombre_csv
    carpeta_csv = folder_download + '/mov_csv/'
    nombre_csv = carpeta_csv + mov_eval + '_' + str(dat) + '.csv'
    if not os.path.exists(carpeta_csv):
        os.mkdir(carpeta_csv)

    if not os.path.exists(nombre_csv):
        camera = cv2.VideoCapture(name_video)
        mp_hands = mp.solutions.hands

        ret = True
        frame = 0
        fps = camera.get(cv2.CAP_PROP_FPS)
        while ret:
            ret, image = camera.read()
            if ret:
                with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
                    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    #REVISAR QUE DEBE HACER SI NO ENCUENTRA MANO
                    if not results.multi_hand_landmarks:
                        continue
                    row = [float(frame)/fps]
                    for finger in results.multi_hand_landmarks[0].landmark:
                        row.append(finger.x)
                        row.append(finger.y)
                        row.append(finger.z)

                    with open(nombre_csv, mode='a') as f:
                        f_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        f_writer.writerow(row)
                frame += 1


def procesar_video(nombre_csv):
    global test_id
    global fig1
    global cursor1
    global plot1
    global t1
    global dedo1
    global dedo2
    global cursor2
    global fig2
    global plot2
    global t2
    global dedo1_seg
    global dedo2_seg
    global cursor3
    global cursor4
    global fig3
    global cursor5
    global cursor6
    global fig4
    global cursor7
    global cursor8
    mov = movimiento.get()
    if mov == 'Golpeteo de dedos':
        mov_eval = 'fingertap'
    elif mov == 'Prono supinacion':
        mov_eval = 'pronosup'
    elif mov == 'Cierre de puño':
        mov_eval = 'fist'

    canvas2.delete(test_id)
    test_id = canvas2.create_text(690.0, 50.0, anchor="nw", text=nombre_csv,
                                  fill="#000000", font=("Roboto", 13 * -1), width=275)

    video_procesado = Parkinson_movements(nombre_csv, mov_eval)
    video_procesado.organize_signal()
    video_procesado.filtered_signal()
    video_procesado.calc_speed()

    t_amp, mov_amp1, amp_trend1, mov_amp2, amp_trend2 = video_procesado.calc_amplitud()

    t1, dedo1, dedo2, t2, dedo1_seg, dedo2_seg, t3, dedo1_vel, dedo2_vel = video_procesado.plotear_graficas()

    fig1 = Figure(figsize=(5, 4), dpi=100)
    plot1 = fig1.add_subplot(111)
    plot1.plot(t1, dedo1)
    plot1.plot(t1, dedo2)
    plot1.set_xlabel('Tiempo (s)')
    plot1.set_ylabel('Amplitud')
    if mov == 'Golpeteo de dedos':
        plot1.set_title('Finger tapping en y - original')
        plot1.legend(['Pulgar', 'Índice'], loc='upper right')
    elif mov == 'Prono supinacion':
        plot1.set_title('Prono-supinación en x - original')
        plot1.legend(['Pulgar', 'Meñique'], loc='upper right')
    elif mov == 'Cierre de puño':
        plot1.set_title('Fist open-close en y - original')
        plot1.legend(['Pulgar', 'Meñique'], loc='upper right')
    fig1.tight_layout()

    canvasf = FigureCanvasTkAgg(fig1, master=tab2)
    toolbar1 = NavigationToolbar2Tk(canvasf, tab2)
    canvasf.draw()
    toolbar1.place(x=160.0, y=540.0)
    canvasf.get_tk_widget().place(x=80.0, y=145.0)
    toolbar1.update()
    # x=80, y=145


    fig2 = Figure(figsize=(5, 4), dpi=100)
    plot2 = fig2.add_subplot(111)
    plot2.plot(t2, dedo1_seg, visible='False')
    plot2.plot(t2, dedo2_seg, visible='False')
    plot2.set_xlabel('Tiempo (s)')
    plot2.set_ylabel('Amplitud')
    if mov == 'Golpeteo de dedos':
        plot2.set_title('Finger tapping en y - segmentada')
        plot2.legend(['Pulgar', 'Índice'], loc='upper right')
    elif mov == 'Prono supinacion':
        plot2.set_title('Prono-supinación en x - segmentada')
        plot2.legend(['Pulgar', 'Meñique'], loc='upper right')
    elif mov == 'Cierre de puño':
        plot2.set_title('Fist open-close en y - segmentada')
        plot2.legend(['Pulgar', 'Meñique'], loc='upper right')

    canvasf = FigureCanvasTkAgg(fig2, master=tab2)
    canvasf.draw()


    # x=80, y=145

    fig3 = Figure(figsize=(5, 4), dpi=100)
    plot3 = fig3.add_subplot(111)
    plot3.plot(t3, dedo1_vel)
    plot3.plot(t3, dedo2_vel)
    cursor5 = SnaptoCursor(plot3, t3, dedo1_vel, "crimson")
    fig3.canvas.mpl_connect('motion_notify_event', cursor5.mouse_move)
    cursor6 = SnaptoCursor(plot3, t3, dedo2_vel, "blue")
    fig3.canvas.mpl_connect('motion_notify_event', cursor6.mouse_move)
    plot3.set_xlabel('Tiempo (s)')
    plot3.set_ylabel('Velocidad')
    if mov == 'Golpeteo de dedos':
        plot3.set_title('Finger tapping en y - velocidad')
        plot3.legend(['Pulgar', 'Índice'], loc='upper right')
    elif mov == 'Prono supinacion':
        plot3.set_title('Prono-supinación en x - velocidad')
        plot3.legend(['Pulgar', 'Meñique'], loc='upper right')
    elif mov == 'Cierre de puño':
        plot3.set_title('Fist open-close en y - velocidad')
        plot3.legend(['Pulgar', 'Meñique'], loc='upper right')
    fig3.tight_layout()

    canvasf = FigureCanvasTkAgg(fig3, master=tab2)
    canvasf.draw()
    toolbar3 = NavigationToolbar2Tk(canvasf,
                                    tab2)
    toolbar3.update()
    toolbar3.place(x=770.0, y=540.0)
    canvasf.get_tk_widget().place(x=690.0, y=145.0)
    # x=680, y=145


    fig4 = Figure(figsize=(5, 4), dpi=100)
    plot4 = fig4.add_subplot(111)
    plot4.plot(t_amp, mov_amp1, 'tab:blue')
    plot4.plot(t_amp, amp_trend1, color='tab:blue', linestyle=':')
    plot4.plot(t_amp, mov_amp2, color='tab:orange')
    plot4.plot(t_amp, amp_trend2, color='tab:orange', linestyle=':')
    cursor7 = SnaptoCursor(plot4, t_amp, mov_amp1, "crimson")
    fig4.canvas.mpl_connect('motion_notify_event', cursor7.mouse_move)
    cursor8 = SnaptoCursor(plot4, t_amp, mov_amp2, "blue")
    fig4.canvas.mpl_connect('motion_notify_event', cursor8.mouse_move)
    plot4.set_title("Tendencias de amplitud de las señales - " + mov)
    plot4.set_xlabel('Tiempo(s)')
    plot4.set_ylabel('Amplitud')
    if mov == 'Golpeteo de dedos':
        plot4.legend(['Pulgar', "Tendencia pulgar", 'Índice', "Tendencia indice"], loc='upper right')
    elif mov == 'Prono supinacion':
        plot4.legend(['Pulgar', "Tendencia pulgar", 'Meñique', "Tendencia meñique"], loc='upper right')
    elif mov == 'Cierre de puño':
        # plot4.set_title("Falta arreglar fist open close en extraccion!!!") #No se si aun falta arreglar, toca revisar
        plot4.legend(['Pulgar', 'Meñique'], loc='upper right')
    fig4.tight_layout()


def cursor_graph1():
    global banderas
    global cursor1
    global cursor2
    global cursor3
    global cursor4
    global cid1
    global cid2
    global cid3
    global cid4
    banderas = not banderas
    if banderas:
        cursor1 = SnaptoCursor(plot1, t1, dedo1, "crimson")
        cid1 = fig1.canvas.mpl_connect('motion_notify_event', cursor1.mouse_move)
        cursor2 = SnaptoCursor(plot1, t1, dedo2, "blue")
        cid2 = fig1.canvas.mpl_connect('motion_notify_event', cursor2.mouse_move)
        cursor3 = SnaptoCursor(plot2, t2, dedo1_seg, "crimson")
        cid3 = fig2.canvas.mpl_connect('motion_notify_event', cursor3.mouse_move)
        cursor4 = SnaptoCursor(plot2, t2, dedo2_seg, "blue")
        cid4 = fig2.canvas.mpl_connect('motion_notify_event', cursor4.mouse_move)
    else:
        fig1.canvas.mpl_disconnect(cid1)
        fig1.canvas.mpl_disconnect(cid2)
        fig2.canvas.mpl_disconnect(cid3)
        fig2.canvas.mpl_disconnect(cid4)
        cursor1.delete_cursor()
        cursor2.delete_cursor()
        cursor3.delete_cursor()
        cursor4.delete_cursor()



def toggle1():
    global toggle1_btn
    global plot1
    global i
    if toggle1_btn:
        canvasf = FigureCanvasTkAgg(fig2, master=tab2)
        canvasf.get_tk_widget().place(x=80.0, y=145.0)
        toolbar2 = NavigationToolbar2Tk(canvasf, tab2)
        toolbar2.update()
        toolbar2.place(x=160.0, y=540.0)
        fig2.tight_layout()
    else:
        canvasf = FigureCanvasTkAgg(fig1, master=tab2)
        canvasf.get_tk_widget().place(x=80.0, y=145.0)
        toolbar1 = NavigationToolbar2Tk(canvasf, tab2)
        toolbar1.place(x=160.0, y=540.0)
        toolbar1.update()
        fig1.tight_layout()
    toggle1_btn = not toggle1_btn


def toggle2():
    global toggle2_btn
    if toggle2_btn:
        canvasf = FigureCanvasTkAgg(fig4, master=tab2)
        canvasf.draw()
        canvasf.get_tk_widget().place(x=690.0, y=145.0)
        toolbar4 = NavigationToolbar2Tk(canvasf,
                                        tab2)
        toolbar4.update()
        toolbar4.place(x=770.0, y=540.0)

    else:
        canvasf = FigureCanvasTkAgg(fig3, master=tab2)
        canvasf.draw()
        canvasf.get_tk_widget().place(x=690.0, y=145.0)
        toolbar3 = NavigationToolbar2Tk(canvasf,
                                        tab2)
        toolbar3.update()
        toolbar3.place(x=770.0, y=540.0)
    toggle2_btn = not toggle2_btn




nombre_csv = None
toggle1_btn = True
toggle2_btn = True
cap = None
video_name = None
img=None
posicion = 0
banderas = False
fig1 = None
fig2 = None
fig3 = None
fig4 = None

# GUI
root = tk.Tk()
root.title("ParkinsonApp")
# setting the window's size
root.geometry("1340x820")
root.configure(bg="#3A7FF6")

tabControl = ttk.Notebook(root)
tab1 = ttk.Frame(tabControl)

# declaring string variable
# for storing id and movement
patient_id = tk.IntVar()
patient_id.set("0")
movimiento = tk.StringVar()
movimiento.set("Seleccione movimiento")
mano = tk.StringVar()
mano.set("Seleccione mano")
fecha = tk.StringVar(root)
fecha.set("Seleccione fecha")


canvas = Canvas(tab1, bg="#3A7FF6", height=720, width=1280, bd=0, highlightthickness=0, relief="ridge")
canvas.place(x=0, y=0)

canvas.create_rectangle(640.0, 0.0, 1280.0, 720.0, fill="#C4C4C4", outline="")

video_label = Label(tab1)
video_label.config(bg="#C4C4C4")
video_label.place(x=800.0, y=60)

#Boton para buscar fechas disponibles del paciente
button_image_video = PhotoImage(file=relative_to_assets("button_1.png"))
button_image_search = PhotoImage(file=relative_to_assets("button_2.png"))
button_image_folder = PhotoImage(file=relative_to_assets("button_3.png"))

canvas.create_text(31.0, 25.0, anchor="nw", text="Seleccionar prueba", fill="#FFFFFF", font=("Roboto Bold", 36 * -1))
canvas.create_text(903.0, 25.0, anchor="nw", text="Video", fill="#000000", font=("Roboto", 24 * -1))

canvas.create_text(55.0, 80.0, anchor="nw", text="Carpeta de descarga", fill="#FFFFFF", font=("Roboto", 24 * -1))
entry_image_folder = PhotoImage(file=relative_to_assets("entry_2.png"))
entry_bg_2 = canvas.create_image(210.0, 140, image=entry_image_folder)

# FOLDER PLACEHOLDER
folder_id = canvas.create_text(60.0, 127.0, anchor="nw", text="Carpeta no seleccionada", fill="#636363",
                               font=("Roboto", 18 * -1), width=270)

#Boton para elegir carpeta
folder_btn = tk.Button(tab1, image=button_image_folder,
                       borderwidth=0,
                       highlightthickness=0,
                       command=carpeta_descarga,
                       relief="flat")
folder_btn.place(x=383.0, y=120.0, width=137.0, height=47.0)

canvas.create_text(55.0, 185.0, anchor="nw", text="ID del paciente", fill="#FFFFFF", font=("Roboto", 24 * -1))
entry_image_id = PhotoImage(file=relative_to_assets("entry_1.png"))
entry_bg_1 = canvas.create_image(210, 245, image=entry_image_id)
patient_id = Entry(tab1, bd=0, bg="#FFFFFF", highlightthickness=0)
patient_id.place(x=62.0, y=222.5, width=275.0, height=45.0)

# Boton para buscar paciente en DB
search_btn = tk.Button(tab1, image=button_image_search,
                       borderwidth=0,
                       highlightthickness=0,
                       command=search_vid,
                       relief="flat")
search_btn.place(x=383.0, y=225.0, width=137.0, height=47.0)


canvas.create_rectangle(170.0, 360.0, 470.0, 588.0, fill="#FFFFFF", outline="")
canvas.create_text(220.0, 320.0, anchor="nw", text="Seleccionar video", fill="#FFFFFF", font=("Roboto", 28 * -1))

# Boton para descargar carpeta y elegir video
video_btn = tk.Button(tab1, image=button_image_video,
                      borderwidth=0,
                      highlightthickness=0,
                      command=adquirir_video,
                      relief="flat")
video_btn.place(x=220.0, y=620.0, width=200.0, height=47.0)

iniciar_btn = tk.Button(tab1, text='Iniciar', command=btn_iniciar)
iniciar_btn.place(x=725.0, y=670.0, width=200.0, height=47.0)

pausar_btn = tk.Button(tab1, text='Pausar', command=btn_pausar)
pausar_btn.place(x=1025.0, y=670.0, width=200.0, height=47.0)


dates_list = ['No hay fechas disponibles']
hand_entry = OptionMenu(tab1, mano, "Derecha", "Izquierda")
movement_entry = OptionMenu(tab1, movimiento, "Golpeteo de dedos", "Prono supinacion", "Cierre de puño")
date_entry = tk.OptionMenu(tab1, fecha, *dates_list)

hand_entry.place(x=220.0, y=390.0, width=200.0, height=47.0)
movement_entry.place(x=220.0, y=430.0, width=200.0, height=47.0)
date_entry.place(x=220.0, y=470.0, width=200.0, height=47.0)


tab2 = ttk.Frame(tabControl)
canvas2 = Canvas(tab2, bg="#3A7FF6", height=720, width=1280, bd=0, highlightthickness=0, relief="ridge")
canvas2.place(x=0, y=0)

button_image_process = PhotoImage(file=relative_to_assets("tab2_button_1.png"))
button_image_graph1 = PhotoImage(file=relative_to_assets("tab2_button_2.png"))
button_image_graph2 = PhotoImage(file=relative_to_assets("tab2_button_3.png"))

#Boton para procesar video y obtener graficas
procesamiento_btn = tk.Button(tab2, image=button_image_process,
                              borderwidth=0,
                              highlightthickness=0,
                              command=lambda: procesar_video(nombre_csv),
                              relief="flat")
procesamiento_btn.place(x=1020.0, y=43.0, width=206.0, height=47.0)

graph1_btn = tk.Button(tab2, image=button_image_graph1,
                       borderwidth=0,
                       highlightthickness=0,
                       command=lambda: toggle1(),
                       relief="flat")
graph1_btn.place(x=395.0, y=600.0, width=206.0, height=47.0)

graph2_btn = tk.Button(tab2, image=button_image_graph2,
                       borderwidth=0,
                       highlightthickness=0,
                       command=lambda: toggle2(),
                       relief="flat")
graph2_btn.place(x=1020.0, y=600.0, width=206.0, height=47.0)

cursor_graph1_btn = tk.Button(tab2, text='Cursor',
                       borderwidth=0,
                       highlightthickness=0,
                       command=lambda: cursor_graph1(),
                       relief="flat")
cursor_graph1_btn.place(x=100, y=600.0, width=206.0, height=47.0)



#Recuadros blancos para ver gráficas
image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
image_1 = canvas2.create_image(337.0, 343.0, image=image_image_1)

image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
image_2 = canvas2.create_image(945.0, 343.0, image=image_image_2)

#Recuadro blanco para ver prueba seleccionada
image_image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
image_3 = canvas2.create_image(846.0, 67.0, image=image_image_3)


test_id = canvas2.create_text(705.0, 57.0, anchor="nw", text="No se ha seleccionado prueba",
                    fill="#636363", font=("Roboto", 18 * -1))

canvas2.create_text(456.0, 54.0, anchor="nw", text="Prueba seleccionada",
                    fill="#FFFFFF", font=("Roboto", 24 * -1))

canvas2.create_text(31.0, 44.0, anchor="nw", text="Analizar prueba",
                    fill="#FFFFFF", font=("Roboto Bold", 36 * -1))


tabControl.add(tab1, text='Video')
tabControl.add(tab2, text='Gráficas')
tabControl.pack(expand=1, fill='both')

# performing an infinite loop for the window to display
root.mainloop()

