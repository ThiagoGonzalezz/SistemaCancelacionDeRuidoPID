import tkinter as tk
from tkinter import ttk, filedialog
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
Ki = 0
Kd = 0

primer_error_derivativo = True
errorAcumulado = 0
cantidadDeMediciones = 0
errorAnterior = 0

try:
    music, _ = librosa.load('haVueltoElMatador.wav', sr=sample_rate, mono=True)
except FileNotFoundError as e:
    print(f"Error: {e}. Asegúrate de que 'haVueltoElMatador.wav' esté en {os.getcwd()}")
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

# Ruido extra cargado por el usuario
extra_noise = np.zeros_like(motor_noise)
extra_noise_amplitude = 0.0
extra_noise_idx = 0  

music_amplitude = 0.0
motor_amplitude = 0.0
horn_amplitude = 0.0
enable_pid = True  # Siempre activo
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

def set_extra_noise_amplitude(value):
    global extra_noise_amplitude
    extra_noise_amplitude = float(value)
    extra_noise_label_var.set(f"Amplitud Ruido Extra: {extra_noise_amplitude:.2f}")

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

def toggle_pause():
    global is_paused, stream
    is_paused = not is_paused
    pause_button.config(text="Reanudar" if is_paused else "Pausar")
    if is_paused:
        stream.stop()
    else:
        stream.start()

# =================== FUNCIONES DE SEÑAL ===================
def generate_signal(t, idx):
    global extra_noise_idx
    idx = idx % len(music)
    end_idx = min(idx + len(t), len(music))
    music_signal = music_amplitude * music[idx:end_idx]
    motor = motor_amplitude * motor_noise[idx:end_idx]
    horn = horn_amplitude * horn_noise[idx:end_idx]
    
    # Manejo del ruido extra de forma cíclica
    extra = np.zeros_like(t)  # Inicializamos con ceros
    if extra_noise_amplitude > 0 and len(extra_noise) > 0:
        for i in range(len(t)):
            extra[i] = extra_noise[extra_noise_idx % len(extra_noise)]
            extra_noise_idx = (extra_noise_idx + 1) % len(extra_noise)
        extra *= extra_noise_amplitude
    
    noise_signal = motor + horn + extra
    if len(music_signal) < len(t):
        music_signal = np.pad(music_signal, (0, len(t) - len(music_signal)), 'constant')
        noise_signal = np.pad(noise_signal, (0, len(t) - len(noise_signal)), 'constant')
    
    return music_signal, noise_signal

# =================== FUNCIONES PID ===================
def controladorPID(error):
    return controladorProporcional(error) + controladorDerivativo(error) + controladorIntegral(error)

def controladorProporcional(error):
    return Kp * error

def controladorIntegral(error):
    global errorAcumulado, cantidadDeMediciones
    errorAcumulado += error
    cantidadDeMediciones += 1
    valorMedio = errorAcumulado / cantidadDeMediciones
    return Ki * valorMedio


def controladorDerivativo(error):
    global errorAnterior, primer_error_derivativo, horn_active_var
    try:
        
        if not isinstance(error, np.ndarray):
            error = np.array([error]) if np.isscalar(error) else np.zeros_like(signals[0])

        if primer_error_derivativo:
            primer_error_derivativo = False
            errorAnterior = np.zeros_like(error) if isinstance(error, np.ndarray) else 0.0
            return np.zeros_like(error)

        pendiente_error = np.mean(error - errorAnterior) / dt if np.any(error - errorAnterior) else 0.0
        errorAnterior = error.copy() 
        
        #acá empieza lo fake
        music_signal = signals[0]  
        noise_signal = signals[1]  

        # Asegurar que noise_signal tenga la misma longitud que music_signal
        if len(noise_signal) < len(music_signal):
            noise_signal = np.pad(noise_signal, (0, len(music_signal) - len(noise_signal)), mode='constant')

        # detección pendiente no nula
        noise_peak = np.max(np.abs(noise_signal)) if np.any(noise_signal) else 0.0
        music_peak = np.max(np.abs(music_signal)) if np.any(music_signal) else 1e-10
        noise_to_music_ratio = noise_peak / music_peak if music_peak > 0 else 0.0

        flag_deriva = (noise_peak > 0.02 and noise_to_music_ratio > 0.8 and 
                          abs(pendiente_error) > 0.02)

        derivative_output = np.zeros_like(music_signal)
        if flag_deriva:
            anti_noise = -noise_signal * 0.2 
            derivative_output = Kd * anti_noise #acá si le ponemos algo de la pendiente del error capaz pasa  
            horn_active_var.set(f"Ruido activo: Sí (Ratio: {noise_to_music_ratio:.2f}, Pendiente: {pendiente_error:.3f})")
        else:
            # si no hay cambio brusco, derivativo nulo
            derivative_output = np.zeros_like(music_signal)
            horn_active_var.set(f"Ruido activo: No (Ratio: {noise_to_music_ratio:.2f}, Pendiente: {pendiente_error:.3f})")

        # Limitamos la salida para lograr estabilidad
        derivative_output = np.clip(derivative_output, -output_max * 0.1, output_max * 0.1)

        return derivative_output

    except Exception as e:
        print(f"Error en controladorDerivativo: {e}")
        return np.zeros_like(signals[0])




















# =================== CALLBACK AUDIO ===================
def audio_callback(outdata, frames, time, status):
    global music_idx, extra_noise_idx, signals, error_rms, retroalimentacion, entradaAnterior
    if is_paused:
        outdata.fill(0)
        return

    error = entradaAnterior - retroalimentacion
    music_signal, noise_signal = generate_signal(t, music_idx)
    antiruido = controladorPID(error)
    entradaAnterior = music_signal
    salida = music_signal + antiruido + noise_signal

    error_rms = np.sqrt(np.mean(error**2))
    outdata[:, 0] = np.clip(salida, -1.0, 1.0)
    music_idx = (music_idx + len(t)) % len(music)
    # Nota: extra_noise_idx se actualiza dentro de generate_signal
    retroalimentacion = salida
    signals = (music_signal, salida, error, antiruido, salida, noise_signal)

# =================== CARGAR AUDIO ===================
def cargar_musica():
    global music, motor_noise, horn_noise, extra_noise, music_idx, primer_error_derivativo
    archivo = filedialog.askopenfilename(filetypes=[("Archivos WAV", "*.wav")])
    if archivo:
        y, sr = librosa.load(archivo, sr=sample_rate, mono=True)
        if len(y) == 0:
            return
        y = y / np.max(np.abs(y))
        music = y
        # Al cambiar música, reajustar ruidos para que tengan la misma longitud
        min_len = len(music)
        motor_noise.resize(min_len, refcheck=False)
        horn_noise.resize(min_len, refcheck=False)
        extra_noise.resize(min_len, refcheck=False)
        music_idx = 0
        primer_error_derivativo = True
        print(f"Música cargada: {archivo}")

def cargar_ruido():
    global extra_noise, extra_noise_amplitude, extra_noise_idx
    archivo = filedialog.askopenfilename(filetypes=[("Archivos WAV", "*.wav")])
    if archivo:
        y, sr = librosa.load(archivo, sr=sample_rate, mono=True)
        if len(y) == 0:
            return
        y = y / np.max(np.abs(y))
        extra_noise = y  # No redimensionamos, permitimos que el ruido extra tenga su propia longitud
        extra_noise_idx = 0  # Reiniciar el índice al cargar nuevo ruido
        if extra_noise_amplitude == 0:
            extra_noise_amplitude = 0.0
            extra_noise_scale.set(extra_noise_amplitude)
        extra_noise_label_var.set(f"Amplitud Ruido Extra: {extra_noise_amplitude:.2f}")
        print(f"Ruido cargado: {archivo}")

# =================== INTERFAZ ===================
root = tk.Tk()
root.title("Simulación ANC PID")

main_frame = ttk.Frame(root)
main_frame.pack(fill="both", expand=True)

left_panel = ttk.Frame(main_frame)
left_panel.pack(side="left", fill="y", padx=10, pady=10)

right_panel = ttk.Frame(main_frame)
right_panel.pack(side="right", fill="both", expand=True)

extra_noise_scale = None

# Controles
music_label_var = tk.StringVar()
motor_label_var = tk.StringVar()
horn_label_var = tk.StringVar()
extra_noise_label_var = tk.StringVar()
kp_label_var = tk.StringVar()
ki_label_var = tk.StringVar()
kd_label_var = tk.StringVar()
error_rms_var = tk.StringVar()
horn_active_var = tk.StringVar(value="Ruido activo: No")

labels = [
    ("Música", set_music_amplitude, music_label_var),
    ("Motor", set_motor_amplitude, motor_label_var),
    ("Bocinazo", set_horn_amplitude, horn_label_var),
    ("Ruido Extra", set_extra_noise_amplitude, extra_noise_label_var),
    ("Kp", set_Kp, kp_label_var),
    ("Ki", set_Ki, ki_label_var),
    ("Kd", set_Kd, kd_label_var),
]

for i, (label, cmd, var) in enumerate(labels):
    scale = ttk.Scale(left_panel, from_=0, to=5 if 'K' in label else 1, orient="horizontal", command=cmd)
    if label == "Ruido Extra":
        scale.set(extra_noise_amplitude)
        extra_noise_scale = scale  # <- Guardamos referencia acá
    else:
        scale.set(0)
    scale.grid(row=i, column=1, sticky="ew", pady=3)
    ttk.Label(left_panel, textvariable=var).grid(row=i, column=0, sticky="w")

# Botones cargar música y ruido
btn_cargar_musica = ttk.Button(left_panel, text="Cargar Música (WAV)", command=cargar_musica)
btn_cargar_musica.grid(row=len(labels), column=0, columnspan=2, pady=5, sticky="ew")

btn_cargar_ruido = ttk.Button(left_panel, text="Cargar Ruido (WAV)", command=cargar_ruido)
btn_cargar_ruido.grid(row=len(labels)+1, column=0, columnspan=2, pady=5, sticky="ew")

# Botón reiniciar integral y pausa
reset_btn = ttk.Button(left_panel, text="Reiniciar Integral", command=reset_integral)
reset_btn.grid(row=len(labels)+2, column=0, columnspan=2, pady=5, sticky="ew")

pause_button = ttk.Button(left_panel, text="Pausar", command=toggle_pause)
pause_button.grid(row=len(labels)+3, column=0, columnspan=2, pady=5, sticky="ew")

ttk.Label(left_panel, textvariable=error_rms_var).grid(row=len(labels)+4, column=0, columnspan=2, pady=10)

# Gráficos principales
fig, axs = plt.subplots(6, 1, figsize=(8, 8), constrained_layout=True)
axes = axs
lines = []
colors = ['blue', 'purple', 'red', 'green', 'brown' ,'orange']
titles = ["Θi - (Música)", "Θo - (Sonido percibido por el usuario)", "e - (Sonido residual que no pudo ser cancelado)", "Θoc - (Señal antirruido)", "f unitaria - (Sonido captado por el micrófono interno)", "P - (Ruido externo)"]
for ax, title, c in zip(axs, titles, colors):
    line, = ax.plot(t, np.zeros_like(t), color=c)
    ax.set_title(title)
    ax.set_xlim(0, block_duration)
    ax.grid(True)
    lines.append(line)
canvas_main = FigureCanvasTkAgg(fig, master=right_panel)
canvas_main.get_tk_widget().pack(fill="both", expand=True)

line_music, line_output, line_error, line_control, line_feedback, line_noise = lines

# Gráficos PID debajo del panel izquierdo
fig_pid, (ax_p, ax_i, ax_d) = plt.subplots(3, 1, figsize=(3, 2.5), dpi=100, constrained_layout=True)
line_p, = ax_p.plot(t, np.zeros_like(t), color="blue")
line_i, = ax_i.plot(t, np.zeros_like(t), color="green")
line_d, = ax_d.plot(t, np.zeros_like(t), color="red")

for ax, title in zip([ax_p, ax_i, ax_d], ["Proporcional", "Integral", "Derivativo"]):
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, block_duration)
    ax.grid(True)

canvas_pid = FigureCanvasTkAgg(fig_pid, master=left_panel)
canvas_pid.get_tk_widget().grid(row=len(labels)+5, column=0, columnspan=2, pady=10)

# =================== ANIMACIÓN ===================
def update_plots(frame):
    global error_rms
    if is_paused:
        return lines + [line_p, line_i, line_d]

    music_signal, output, error, anti_noise, retroalimentacion, noise_signal = signals
    proportional = np.full_like(t, controladorProporcional(error))
    integral = np.full_like(t, controladorIntegral(error))
    derivative = np.full_like(t, controladorDerivativo(error))

    line_music.set_ydata(music_signal)
    line_output.set_ydata(output)
    line_error.set_ydata(error)
    line_control.set_ydata(anti_noise)
    line_feedback.set_ydata(retroalimentacion)
    line_noise.set_ydata(noise_signal)
    line_p.set_ydata(proportional)
    line_i.set_ydata(integral)
    line_d.set_ydata(derivative)

    for ax, signal in zip(axes, signals):
        max_val = max(np.max(np.abs(signal)) * 1.5, 0.1)
        ax.set_ylim(-max_val, max_val)

    for ax, signal in zip([ax_p, ax_i, ax_d], [proportional, integral, derivative]):
        max_val = max(np.max(np.abs(signal)) * 1.5, 0.1)
        ax.set_ylim(-max_val, max_val)

    error_rms_var.set(f"Error RMS: {error_rms:.4f}")
    return lines + [line_p, line_i, line_d]

ani = FuncAnimation(fig, update_plots, interval=block_duration * 1000, blit=False)

stream = sd.OutputStream(samplerate=sample_rate, channels=1, callback=audio_callback, blocksize=len(t))
stream.start()

root.mainloop()

stream.stop()
stream.close()
