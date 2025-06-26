import tkinter as tk
from tkinter import ttk
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import librosa
import os


# Configuración inicial
sample_rate = 44100
block_duration = 0.02  # Reducido para fluidez
t = np.linspace(0, block_duration, int(sample_rate * block_duration))
dt = t[1] - t[0]
retroalimentacion = 0
entradaAnterior = 0

#VARIABLES GLOBALES CONTROLADORES
Kp = 0  # Ganancia proporcional
Ki = 0.8    # Ganancia integral
Kd = 0   # Ganancia derivativa

#Integral
errorAcumulado = 0
cantidadDeMediciones = 0

#Derovativo
errorAnterior = 0 

# Cargar música
try:
    music, _ = librosa.load('music.wav', sr=sample_rate, mono=True)
except FileNotFoundError as e:
    print(f"Error: {e}. Asegúrate de que 'music.wav' esté en {os.getcwd()}")
    exit(1)

# Normalizar música
music = music / np.max(np.abs(music))

# Perturbaciones (acá podemos poner audios de ruidos reales pero cuando lo hice se me bugeó)
min_length = len(music)
motor_noise = 0.4 * np.sin(2 * np.pi * 200 * np.linspace(0, min_length/sample_rate, min_length))
horn_noise = np.zeros(min_length)
pulse_duration = int(0.5 * sample_rate)
pulse_interval = int(2 * sample_rate)
for i in range(0, min_length, pulse_interval):
    end = min(i + pulse_duration, min_length)
    horn_noise[i:end] = 0.4 * np.sin(2 * np.pi * 1000 * np.linspace(0, (end-i)/sample_rate, end-i))
motor_noise = motor_noise / np.max(np.abs(motor_noise))
horn_noise = horn_noise / np.max(np.abs(horn_noise))

# 
music_amplitude = 0.0
motor_amplitude = 0.0
horn_amplitude = 0.0
enable_pid = False  # PID. Se podría separar en los 3 controladores
music_idx = 0
signals = (np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), np.zeros_like(t))
is_paused = False
stream = None
error_rms = 0.0
output_max = 0.5  # Límite para evitar saturación

# Funciones para actualizar amplitudes
def set_music_amplitude(value):
    global music_amplitude
    music_amplitude = float(value)
    music_label_var.set(f"Amplitud Música: {music_amplitude:.2f}")

def set_motor_amplitude(value):
    global motor_amplitude
    motor_amplitude = float(value)
    motor_label_var.set(f"Amplitud Motor: {motor_amplitude:.2f}")

def set_horn_amplitude(value):
    global horn_amplitude
    horn_amplitude = float(value)
    horn_label_var.set(f"Amplitud Bocinazo: {horn_amplitude:.2f}")

# Generar señales
def generate_signal(t, idx):
    idx = idx % len(music)
    end_idx = min(idx + len(t), len(music))
    music_signal = music_amplitude * music[idx:end_idx]
    noise_signal = motor_amplitude * motor_noise[idx:end_idx] + horn_amplitude * horn_noise[idx:end_idx]
    if len(music_signal) < len(t):
        music_signal = np.pad(music_signal, (0, len(t) - len(music_signal)), 'constant')
        noise_signal = np.pad(noise_signal, (0, len(t) - len(noise_signal)), 'constant')
    return music_signal, noise_signal


#######################################################################
#######################################################################
#######################################################################

# Controlador PID simplificado (fase inversa directa)
# capaz estamos robando con esto pero sino es un quilombo
#def pid_controller(noise_signal):
#    if enable_pid:
#       return np.clip(-noise_signal, -output_max, output_max)  # Fase inversa
#    return np.zeros_like(noise_signal)


def controladorPID(error):
        return controladorProporcional(error) + controladorDerivativo(error) + controladorIntegral(error)
def controladorProporcional(error):
    return Kp * error

def controladorDerivativo(error):
    global errorAnterior

    pendienteError = (error - errorAnterior) / dt
    errorAnterior = error
    return pendienteError * Kd

def controladorIntegral(error):
    global errorAcumulado, cantidadDeMediciones


    errorAcumulado += error
    cantidadDeMediciones += 1
    valorMedio = errorAcumulado / cantidadDeMediciones
    return Ki * valorMedio

# Creador de fase inversa directa)
def invertir_ruido(noise_signal):
    if enable_pid:
        return np.clip(-noise_signal, -output_max, output_max)  # Fase inversa
    return np.zeros_like(noise_signal)

#######################################################################
#######################################################################
#######################################################################

#######################################################################
#######################################################################
#######################################################################


#def audio_callback(outdata, frames, time, status):
#    global music_idx, signals, error_rms
#    if is_paused:
#        outdata.fill(0)
#        return
#    music_signal, noise_signal = generate_signal(t, music_idx)
#    anti_noise = invertir_ruido(noise_signal)
#    error = music_signal - (noise_signal + anti_noise)
#    output = music_signal + noise_signal + anti_noise
#    error_rms = np.sqrt(np.mean(error**2))
#    signals = (music_signal, noise_signal, error, anti_noise, output)
#    outdata[:, 0] = np.clip(output, -1.0, 1.0)
#    music_idx = (music_idx + len(t)) % len(music)

def audio_callback(outdata, frames, time, status):
    global music_idx, signals, error_rms, retroalimentacion, entradaAnterior
    if is_paused:
        outdata.fill(0)
        return
    
    entrada = music
    
    error = entradaAnterior - retroalimentacion
    
    music_signal, noise_signal = generate_signal(t, music_idx)

    #antiruido = invertir_ruido(controladorPID(error))

    antiruido = controladorPID(error)

    entradaAnterior = music_signal

    salida = music_signal + antiruido + noise_signal 

    #Reproducir y actualizar señales
    ############################################################

    error_rms = np.sqrt(np.mean(error**2))

    signals = (music_signal, noise_signal, error, antiruido, salida)
    outdata[:, 0] = np.clip(salida, -1.0, 1.0)
    music_idx = (music_idx + len(t)) % len(music)

    #############################################################


    retroalimentacion = salida

#######################################################################
#######################################################################
#######################################################################

# Pausar/Reanudar
def toggle_pause():
    global is_paused, stream
    is_paused = not is_paused
    pause_button.config(text="Reanudar" if is_paused else "Pausar")
    if is_paused:
        stream.stop()
    else:
        stream.start()

# Activar/Desactivar PID
def toggle_pid():
    global enable_pid
    enable_pid = pid_var.get()

# Configuración de la GUI
root = tk.Tk()
root.title("Simulación ANC")

# Frame para deslizadores
frame_sliders = ttk.Frame(root)
frame_sliders.pack(pady=10)
# Variables dinámicas para mostrar valores PID
kp_label_var = tk.StringVar(value=f"Kp: {Kp:.2f}")
ki_label_var = tk.StringVar(value=f"Ki: {Ki:.2f}")
kd_label_var = tk.StringVar(value=f"Kd: {Kd:.2f}")

def set_Kp(value):
    global Kp
    Kp = float(value)
    kp_label_var.set(f"Kp: {Kp:.2f}")

def set_Ki(value):
    global Ki
    Ki = float(value)
    ki_label_var.set(f"Ki: {Ki:.2f}")

def set_Kd(value):
    global Kd
    Kd = float(value)
    kd_label_var.set(f"Kd: {Kd:.2f}")

# Sliders PID
kp_scale = ttk.Scale(frame_sliders, from_=0, to=5, orient="horizontal", command=set_Kp)
kp_scale.set(Kp)
kp_scale.grid(row=3, column=1, padx=5)
ttk.Label(frame_sliders, textvariable=kp_label_var).grid(row=3, column=0)

ki_scale = ttk.Scale(frame_sliders, from_=0, to=5, orient="horizontal", command=set_Ki)
ki_scale.set(Ki)
ki_scale.grid(row=4, column=1, padx=5)
ttk.Label(frame_sliders, textvariable=ki_label_var).grid(row=4, column=0)

kd_scale = ttk.Scale(frame_sliders, from_=0, to=5, orient="horizontal", command=set_Kd)
kd_scale.set(Kd)
kd_scale.grid(row=5, column=1, padx=5)
ttk.Label(frame_sliders, textvariable=kd_label_var).grid(row=5, column=0)

# Etiquetas dinámicas
music_label_var = tk.StringVar(value=f"Amplitud Música: {music_amplitude:.2f}")
motor_label_var = tk.StringVar(value=f"Amplitud Motor: {motor_amplitude:.2f}")
horn_label_var = tk.StringVar(value=f"Amplitud Bocinazo: {horn_amplitude:.2f}")
error_rms_var = tk.StringVar(value="Error RMS: 0.00")

# Deslizadores
music_scale = ttk.Scale(frame_sliders, from_=0, to=1, orient="horizontal", command=set_music_amplitude)
music_scale.set(0.0)
music_scale.grid(row=0, column=1, padx=5)
ttk.Label(frame_sliders, textvariable=music_label_var).grid(row=0, column=0)

motor_scale = ttk.Scale(frame_sliders, from_=0, to=1, orient="horizontal", command=set_motor_amplitude)
motor_scale.set(0.0)
motor_scale.grid(row=1, column=1, padx=5)
ttk.Label(frame_sliders, textvariable=motor_label_var).grid(row=1, column=0)

horn_scale = ttk.Scale(frame_sliders, from_=0, to=1, orient="horizontal", command=set_horn_amplitude)
horn_scale.set(0.0)
horn_scale.grid(row=2, column=1, padx=5)
ttk.Label(frame_sliders, textvariable=horn_label_var).grid(row=2, column=0)

# Frame para controles
frame_controls = ttk.Frame(root)
frame_controls.pack(pady=10)

# Casilla PID
pid_var = tk.BooleanVar(value=False)
ttk.Checkbutton(frame_controls, text="Activar PID", variable=pid_var, command=toggle_pid).grid(row=0, column=0, padx=5)

# Error RMS
ttk.Label(frame_controls, textvariable=error_rms_var).grid(row=1, column=0, pady=5)

# Botón de pausa
pause_button = ttk.Button(frame_controls, text="Pausar", command=toggle_pause)
pause_button.grid(row=0, column=1, padx=5)

# Gráficas
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 8))
line_music, = ax1.plot(t, np.zeros_like(t), label="Música", color='blue', linewidth=2)
line_noise, = ax2.plot(t, np.zeros_like(t), label="Ruido", color='red', linewidth=2)
line_error, = ax3.plot(t, np.zeros_like(t), label="Error", color='green', linewidth=2)
line_control, = ax4.plot(t, np.zeros_like(t), label="Antirruido", color='purple', linewidth=2)
line_output, = ax5.plot(t, np.zeros_like(t), label="Salida Percibida", color='orange', linewidth=2)
for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, block_duration)
    ax.set_xlabel("Tiempo (s)")
ax1.set_title("Señal de Música")
ax2.set_title("Ruido (Motor + Bocinazo)")
ax3.set_title("Error (Música - Ruido - Antirruido)")
ax4.set_title("Señal Antirruido")
ax5.set_title("Salida Percibida")
plt.tight_layout()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Actualizar gráficas
def update_plots(frame):
    global error_rms
    if is_paused:
        return line_music, line_noise, line_error, line_control, line_output
    music_signal, noise_signal, error, anti_noise, output = signals
    line_music.set_ydata(music_signal)
    line_noise.set_ydata(noise_signal)
    line_error.set_ydata(error)
    line_control.set_ydata(anti_noise)
    line_output.set_ydata(output)
    for ax, signal in zip([ax1, ax2, ax3, ax4, ax5], signals):
        max_val = max(np.max(np.abs(signal)) * 1.5, 0.1)  # Evitar escala cero
        ax.set_ylim(-max_val, max_val)
    error_rms_var.set(f"Error RMS: {error_rms:.4f}")
    return line_music, line_noise, line_error, line_control, line_output

ani = FuncAnimation(fig, update_plots, interval=block_duration*1000, blit=True, save_count=100)

# Iniciar audio
stream = sd.OutputStream(samplerate=sample_rate, channels=1, callback=audio_callback, blocksize=len(t))
stream.start()

# Iniciar GUI
root.mainloop()

# Detener stream
stream.stop()
stream.close()
