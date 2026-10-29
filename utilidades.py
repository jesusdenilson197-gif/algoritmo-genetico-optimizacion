import numpy as np
import matplotlib.pyplot as plt

# 1. VALIDACIÓN MEDIANTE BÚSQUEDA EXHAUSTIVA / FUERZA BRUTA
def validacion_fuerza_bruta(low=-1.0, high=2.0, pasos=100000):
    x_vals = np.linspace(low, high, pasos)
    y_vals = x_vals * np.sin(10 * np.pi * x_vals) + 1.0
    idx_max = np.argmax(y_vals)
    return x_vals[idx_max], y_vals[idx_max]

# 2. GENERACIÓN Y GUARDADO DE LA GRÁFICA DE CONVERGENCIA
def graficar_convergencia(historial):
    plt.figure(figsize=(8, 5))
    plt.plot(historial, color='blue', linewidth=2, label='Mejor Aptitud')
    plt.title('Gráfica de Convergencia del Algoritmo Genético')
    plt.xlabel('Generación')
    plt.ylabel('Aptitud (Fitness)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('grafica_convergencia.png')
    plt.show()