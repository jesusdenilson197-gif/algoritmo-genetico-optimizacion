import json
import random
import math
import numpy as np
import matplotlib.pyplot as plt

def funcion_aptitud(x):
    return x * math.sin(10 * math.pi * x) + 1.0

class AlgoritmoGenetico:
    def __init__(self, config):
        self.pop_size = config["tamano_poblacion"]
        self.generations = config["numero_generaciones"]
        self.p_cruce = config["probabilidad_cruce"]
        self.p_mut = config["probabilidad_mutacion"]
        self.low = config["limite_inferior"]
        self.high = config["limite_superior"]
        
        if config.get("semilla_aleatoria") is not None:
            random.seed(config["semilla_aleatoria"])
            np.random.seed(config["semilla_aleatoria"])

    def crear_poblacion_inicial(self):
        return [random.uniform(self.low, self.high) for _ in range(self.pop_size)]

    def seleccion_torneo(self, poblacion, aptitudes, k=3):
        seleccionados = random.sample(list(zip(poblacion, aptitudes)), k)
        seleccionados.sort(key=lambda x: x[1], reverse=True)
        return seleccionados[0][0]

    def cruce_aritmetico(self, padre1, padre2):
        if random.random() < self.p_cruce:
            alpha = random.random()
            hijo1 = alpha * padre1 + (1 - alpha) * padre2
            hijo2 = (1 - alpha) * padre1 + alpha * padre2
            return hijo1, hijo2
        return padre1, padre2

    def mutacion_gaussiana(self, individuo):
        if random.random() < self.p_mut:
            sigma = (self.high - self.low) * 0.1
            individuo += random.gauss(0, sigma)
            individuo = max(self.low, min(self.high, individuo))
        return individuo

    def ejecutar(self):
        poblacion = self.crear_poblacion_inicial()
        historial_mejor_aptitud = []
        mejor_global_ind = None
        mejor_global_fit = float('-inf')

        for gen in range(self.generations):
            aptitudes = [funcion_aptitud(ind) for ind in poblacion]
            
            max_fit = max(aptitudes)
            idx_max = aptitudes.index(max_fit)
            historial_mejor_aptitud.append(max_fit)

            if max_fit > mejor_global_fit:
                mejor_global_fit = max_fit
                mejor_global_ind = poblacion[idx_max]

            nueva_poblacion = []
            while len(nueva_poblacion) < self.pop_size:
                p1 = self.seleccion_torneo(poblacion, aptitudes)
                p2 = self.seleccion_torneo(poblacion, aptitudes)
                h1, h2 = self.cruce_aritmetico(p1, p2)
                nueva_poblacion.append(self.mutacion_gaussiana(h1))
                if len(nueva_poblacion) < self.pop_size:
                    nueva_poblacion.append(self.mutacion_gaussiana(h2))

            poblacion = nueva_poblacion

        return mejor_global_ind, mejor_global_fit, historial_mejor_aptitud

def validacion_fuerza_bruta(low=-1.0, high=2.0, pasos=100000):
    x_vals = np.linspace(low, high, pasos)
    y_vals = x_vals * np.sin(10 * np.pi * x_vals) + 1.0
    idx_max = np.argmax(y_vals)
    return x_vals[idx_max], y_vals[idx_max]

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

def cargar_configuracion():
    with open("config.json", "r") as f:
        return json.load(f)

def main():
    config = cargar_configuracion()
    
    while True:
        print("\n=== MENÚ INTERACTIVO - ALGORITMO GENÉTICO ===")
        print(f"1. Ejecutar AG con config actual (Población: {config['tamano_poblacion']}, Gen: {config['numero_generaciones']})")
        print("2. Modificar parámetros de configuración")
        print("3. Validar con Fuerza Bruta (Óptimo Global)")
        print("4. Salir")
        
        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            ag = AlgoritmoGenetico(config)
            mejor_ind, mejor_fit, historial = ag.ejecutar()
            print("\n--- RESULTADOS DEL ALGORITMO GENÉTICO ---")
            print(f"Mejor Individuo (x): {mejor_ind:.6f}")
            print(f"Mejor Aptitud f(x): {mejor_fit:.6f}")
            graficar_convergencia(historial)

        elif opcion == "2":
            config["tamano_poblacion"] = int(input("Nuevo tamaño de población: "))
            config["numero_generaciones"] = int(input("Nuevo número de generaciones: "))
            config["probabilidad_cruce"] = float(input("Probabilidad de cruce (0-1): "))
            config["probabilidad_mutacion"] = float(input("Probabilidad de mutación (0-1): "))
            
            with open("config.json", "w") as f:
                json.dump(config, f, indent=4)
            print("¡Configuración guardada en config.json!")

        elif opcion == "3":
            x_opt, f_opt = validacion_fuerza_bruta()
            print("\n--- VALIDACIÓN (BÚSQUEDA EXHAUSTIVA) ---")
            print(f"Óptimo Teórico/Fuerza Bruta (x): {x_opt:.6f}")
            print(f"Aptitud Máxima f(x): {f_opt:.6f}")

        elif opcion == "4":
            print("Saliendo del programa...")
            break
        else:
            print("Opción inválida. Intente de nuevo.")

if __name__ == "__main__":
    main()
