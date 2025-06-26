import tkinter as tk
from tkinter import ttk
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import librosa
import os

# =================== CONFIGURACIÓN INICIAL ===================
sample_rate = 44100
block_duration = 0.02
t = np.linspace(0, block_duration, int(sample_rate * block_duration))
dt = t[1] - t[0]
retroalimentacion = 0
entradaAnterior = 0

Kp = 0
Ki = 0.8
Kd = 0

errorAcumulado = 0
cantidadDeMediciones = 0
errorAnterior = 0

try:
    music, _ = librosa.load('music.wav', sr=sample_rate, mono=True)
except FileNotFoundError as e:
    print(f"Error: {e}. Asegúrate de que 'music.wav' esté en {os.getcwd()}")
    exit(1)

music = music / np.max(np.abs(music))

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

music_amplitude = 0.0
motor_amplitude = 0.0
horn_amplitude = 0.0
enable_pid = False
music_idx = 0
signals = (np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), np.zeros_like(t))
is_paused = False
stream = None
error_rms = 0.0
output_max = 0.5

# =================== FUNCIONES DE CONTROL ===================
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

def reset_integral():
    global errorAcumulado, cantidadDeMediciones
    errorAcumulado = 0
    cantidadDeMediciones = 0

# =================== FUNCIONES DE SEÑAL ===================
def generate_signal(t, idx):
    idx = idx % len(music)
    end_idx = min(idx + len(t), len(music))
    music_signal = music_amplitude * music[idx:end_idx]
    noise_signal = motor_amplitude * motor_noise[idx:end_idx] + horn_amplitude * horn_noise[idx:end_idx]
    if len(music_signal) < len(t):
        music_signal = np.pad(music_signal, (0, len(t) - len(music_signal)), 'constant')
        noise_signal = np.pad(noise_signal, (0, len(t) - len(noise_signal)), 'constant')
    return music_signal, noise_signal

# =================== FUNCIONES PID ===================
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

# =================== CALLBACK AUDIO ===================
def audio_callback(outdata, frames, time, status):
    global music_idx, signals, error_rms, retroalimentacion, entradaAnterior
    if is_paused:
        outdata.fill(0)
        return

    error = entradaAnterior - retroalimentacion
    music_signal, noise_signal = generate_signal(t, music_idx)
    antiruido = controladorPID(error)
    entradaAnterior = music_signal
    salida = music_signal + antiruido + noise_signal

    error_rms = np.sqrt(np.mean(error**2))
    signals = (music_signal, noise_signal, error, antiruido, salida)
    outdata[:, 0] = np.clip(salida, -1.0, 1.0)
    music_idx = (music_idx + len(t)) % len(music)
    retroalimentacion = salida

# =================== INTERFAZ ===================
def configurar_estilos():
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TFrame", background="#f0f4f7")
    style.configure("TLabel", background="#f0f4f7", font=("Segoe UI", 10))
    style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
    style.configure("Horizontal.TScale", troughcolor="#e0e0e0", background="#4a90e2")
    style.configure("TCheckbutton", background="#f0f4f7", font=("Segoe UI", 10))

def toggle_pause():
    global is_paused, stream
    is_paused = not is_paused
    pause_button.config(text="Reanudar" if is_paused else "Pausar")
    if is_paused:
        stream.stop()
    else:
        stream.start()

def toggle_pid():
    global enable_pid
    enable_pid = pid_var.get()

root = tk.Tk()
root.title("Simulación ANC")
configurar_estilos()

main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True)

frame_sliders = ttk.Frame(main_frame)
frame_sliders.grid(row=0, column=0, padx=10, pady=10, sticky="n")

music_label_var = tk.StringVar(value=f"Amplitud Música: {music_amplitude:.2f}")
motor_label_var = tk.StringVar(value=f"Amplitud Motor: {motor_amplitude:.2f}")
horn_label_var = tk.StringVar(value=f"Amplitud Bocinazo: {horn_amplitude:.2f}")
kp_label_var = tk.StringVar(value=f"Kp: {Kp:.2f}")
ki_label_var = tk.StringVar(value=f"Ki: {Ki:.2f}")
kd_label_var = tk.StringVar(value=f"Kd: {Kd:.2f}")

crear_slider = lambda text_var, command, row, to_val=1: (
    ttk.Label(frame_sliders, textvariable=text_var).grid(row=row, column=0, sticky="w", padx=10, pady=2),
    ttk.Scale(frame_sliders, from_=0, to=to_val, orient="horizontal", command=command, length=200).grid(row=row, column=1, padx=10, pady=2)
)
crear_slider(music_label_var, set_music_amplitude, 0)
crear_slider(motor_label_var, set_motor_amplitude, 1)
crear_slider(horn_label_var, set_horn_amplitude, 2)
crear_slider(kp_label_var, set_Kp, 3, 5)
crear_slider(ki_label_var, set_Ki, 4, 5)
crear_slider(kd_label_var, set_Kd, 5, 5)

frame_controls = ttk.Frame(frame_sliders)
frame_controls.grid(row=6, column=0, columnspan=2, pady=10)

pid_var = tk.BooleanVar(value=False)
error_rms_var = tk.StringVar(value="Error RMS: 0.00")
pause_button = ttk.Button(frame_controls, text="Pausar", command=toggle_pause)
reset_button = ttk.Button(frame_controls, text="Reiniciar Controlador Integral", command=reset_integral)

ttk.Checkbutton(frame_controls, text="Activar PID", variable=pid_var, command=toggle_pid).grid(row=0, column=0, padx=10, sticky="w")
ttk.Label(frame_controls, textvariable=error_rms_var).grid(row=1, column=0, padx=10, pady=5, sticky="w")
pause_button.grid(row=0, column=1, padx=10)
reset_button.grid(row=1, column=1, padx=10)

# Gráficas
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 8), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=main_frame)
canvas.get_tk_widget().grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

main_frame.columnconfigure(1, weight=1)
main_frame.rowconfigure(0, weight=1)

line_music, = ax1.plot(t, np.zeros_like(t), label="Música", color='blue', linewidth=2)
line_noise, = ax2.plot(t, np.zeros_like(t), label="Ruido", color='red', linewidth=2)
line_error, = ax3.plot(t, np.zeros_like(t), label="Error", color='green', linewidth=2)
line_control, = ax4.plot(t, np.zeros_like(t), label="Antirruido", color='purple', linewidth=2)
line_output, = ax5.plot(t, np.zeros_like(t), label="Salida Percibida", color='orange', linewidth=2)

for ax, title in zip([ax1, ax2, ax3, ax4, ax5],
                     ["Señal de Música", "Ruido (Motor + Bocinazo)", "Error (Música - Ruido - Antirruido)",
                      "Señal Antirruido", "Salida Percibida"]):
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(0, block_duration)
    ax.set_xlabel("Tiempo (s)")
    ax.set_title(title, fontweight="bold")

plt.tight_layout()

# =================== ANIMACIÓN ===================
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
        max_val = max(np.max(np.abs(signal)) * 1.5, 0.1)
        ax.set_ylim(-max_val, max_val)
    error_rms_var.set(f"Error RMS: {error_rms:.4f}")
    return line_music, line_noise, line_error, line_control, line_output

ani = FuncAnimation(fig, update_plots, interval=block_duration*1000, blit=True, save_count=100)

stream = sd.OutputStream(samplerate=sample_rate, channels=1, callback=audio_callback, blocksize=len(t))
stream.start()
root.mainloop()
stream.stop()
stream.close()
