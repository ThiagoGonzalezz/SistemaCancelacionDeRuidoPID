# 🎧 Simulador de Cancelación Activa de Ruido con Control PID y Feedforward

**Trabajo Práctico Integrador**

- **Materia**: Teoría de Control  
- **Año**: 2025  
- **Profesor**: Omar Oscár Civale
- **Alumnos**: Agustín Gabriel Podhainy Vignola, Thiago Martín González

---

## 🛠 Requisitos

Antes de ejecutar el simulador, asegurate de cumplir con los siguientes requisitos:

### Sistema

- Computadora con **Windows**, **Linux** o **macOS**
- **Python 3.10** o superior instalado
  👉 [¿No tenés Python? Instalalo desde esta guía](https://www.desdecero.dev/python/como-instalar-python/)


### Bibliotecas necesarias

Se deben instalar las siguientes bibliotecas de Python:

- `numpy`
- `sounddevice`
- `matplotlib`
- `librosa`
- `tkinter` (incluido por defecto en instalaciones estándar de Python)

---

## ⚙️ Instalación

### 1. Clonar o descargar el repositorio

```bash
git clone https://github.com/ThiagoGonzalezz/SistemaCancelacionDeRuidoPID.git
```

O descargar el ZIP desde GitHub y descomprimirlo.

### 2. Instalar dependencias

Desde una terminal ubicada en la carpeta del proyecto, ejecutar:

```bash
pip install numpy sounddevice matplotlib librosa
```

> ⚠️ En Windows, puede que necesites abrir el "Símbolo del sistema" como administrador.

---

## ▶️ Ejecución

Desde la terminal, dentro del directorio del proyecto, correr:

```bash
python simulacionTc.py
```

Esto abrirá la interfaz gráfica del simulador.

---

## 🎛 Funcionalidades del simulador

- Reproducción de una señal musical como señal deseada.
- Introducción de perturbaciones externas: ruido de motor (200 Hz) y bocina (1000 Hz).
- Introducción de perturbación interna (crosstalk – 500 Hz).
- Botón para cargar una nueva música (debe ser un archivo `.wav`).
- Botón para reiniciar la componente integral del PID.
- Botón para pausar todos los sonidos.
- Activación y desactivación individual de perturbaciones.
- Slider para ajustar la altura de los gráficos y facilitar el análisis visual.
- Controladores disponibles:
  - **PID**: configurable en tiempo real (Kp, Ki, Kd).
  - **Feedforward**: anticipación basada en micrófono externo simulado.
- Visualización gráfica en tiempo real de:
  - Señal deseada
  - Señal de salida
  - Señal de error
  - Señal de control PID
  - Componentes individuales del PID (P, I, D)
  - Señal de retroalimentación unitaria
  - Perturbación total
  - Perturbaciones internas
  - Perturbaciones externas
  - Señal de control feedforward

---

## 📁 Archivos importantes

- `simulacionTc.py`: código principal del simulador.
- `haVueltoElMatador.wav`: señal musical base.
- `bombonAsesino.wav`: señal musical alternativa.
- `crosstalk.wav`: perturbación interna.
- (Podés agregar otros archivos `.wav` si lo deseás)

---

## ℹ️ Notas

- Se recomienda utilizar **auriculares** para percibir mejor el efecto de la cancelación activa.
- En sistemas operativos distintos de Windows, puede ser necesario configurar permisos de audio.
- Si el simulador no reproduce audio o lanza errores, revisar si `sounddevice` detecta correctamente la salida de audio del sistema.

