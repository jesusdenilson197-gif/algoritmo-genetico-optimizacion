import random
import math
import numpy as np

# 1. FUNCIÓN DE APTITUD (FITNESS)
def funcion_aptitud(x):
    """
    f(x) = x * sin(10 * pi * x) + 1
    Maximizar en [-1, 2]
    """
    return x * math.sin(10 * math.pi * x) + 1.0

# 2. CLASE DEL ALGORITMO GENÉTICO Y OPERADORES
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
            # Mantener dentro de los límites
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