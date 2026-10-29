import json
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Importar los módulos desarrollados por tu equipo
from operadores import AlgoritmoGenetico
from utilidades import validacion_fuerza_bruta

class AplicacionAG:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimización mediante Algoritmo Genético - Unidad 3")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)

        # Cargar configuración inicial
        self.config = self.cargar_configuracion()

        # Configuración de estilos
        style = ttk.Style()
        style.theme_use("clam")

        self.crear_interfaz()

    def cargar_configuracion(self):
        try:
            with open("config.json", "r") as f:
                return json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar config.json: {e}")
            return {
                "tamano_poblacion": 100,
                "numero_generaciones": 100,
                "probabilidad_cruce": 0.8,
                "probabilidad_mutacion": 0.05,
                "semilla_aleatoria": 42,
                "limite_inferior": -1.0,
                "limite_superior": 2.0
            }

    def guardar_configuracion(self):
        try:
            self.config["tamano_poblacion"] = int(self.entries["tamano_poblacion"].get())
            self.config["numero_generaciones"] = int(self.entries["numero_generaciones"].get())
            self.config["probabilidad_cruce"] = float(self.entries["probabilidad_cruce"].get())
            self.config["probabilidad_mutacion"] = float(self.entries["probabilidad_mutacion"].get())
            
            with open("config.json", "w") as f:
                json.dump(self.config, f, indent=4)
            messagebox.showinfo("Éxito", "¡Configuración guardada correctamente en config.json!")
        except ValueError:
            messagebox.showerror("Error de Entrada", "Por favor ingresa valores numéricos válidos.")

    def crear_interfaz(self):
        # Panel Superior: Título
        frame_titulo = ttk.Frame(self.root, padding=10)
        frame_titulo.pack(side=tk.TOP, fill=tk.X)
        
        lbl_titulo = ttk.Label(
            frame_titulo, 
            text="Algoritmo Genético: Maximización de f(x) = x · sin(10πx) + 1", 
            font=("Helvetica", 14, "bold")
        )
        lbl_titulo.pack()

        # Panel Izquierdo: Controles y Parámetros
        frame_izq = ttk.LabelFrame(self.root, text=" Configuración de Parámetros ", padding=15)
        frame_izq.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.entries = {}
        campos = [
            ("Tamaño Población:", "tamano_poblacion"),
            ("Número Generaciones:", "numero_generaciones"),
            ("Prob. Cruce (0-1):", "probabilidad_cruce"),
            ("Prob. Mutación (0-1):", "probabilidad_mutacion")
        ]

        for i, (label_text, key) in enumerate(campos):
            lbl = ttk.Label(frame_izq, text=label_text, font=("Helvetica", 10))
            lbl.grid(row=i, column=0, sticky=tk.W, pady=5)
            
            entry = ttk.Entry(frame_izq, width=12)
            entry.insert(0, str(self.config.get(key, "")))
            entry.grid(row=i, column=1, pady=5, padx=5)
            self.entries[key] = entry

        # Botones de Acción
        btn_guardar = ttk.Button(frame_izq, text="💾 Guardar Configuración", command=self.guardar_configuracion)
        btn_guardar.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=8)

        btn_ejecutar = ttk.Button(frame_izq, text="▶ Ejecutar Algoritmo Genético", command=self.ejecutar_ag)
        btn_ejecutar.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=5)

        btn_validar = ttk.Button(frame_izq, text="🔍 Validar con Fuerza Bruta", command=self.ejecutar_validacion)
        btn_validar.grid(row=6, column=0, columnspan=2, sticky=tk.EW, pady=5)

        # Área de Reporte/Resultados
        frame_resultados = ttk.LabelFrame(frame_izq, text=" Resultados ", padding=10)
        frame_resultados.grid(row=7, column=0, columnspan=2, sticky=tk.NSEW, pady=15)

        self.txt_resultados = tk.Text(frame_resultados, height=12, width=32, font=("Consolas", 9))
        self.txt_resultados.pack(fill=tk.BOTH, expand=True)

        # Panel Derecho: Gráfica de Convergencia
        self.frame_derecho = ttk.LabelFrame(self.root, text=" Gráfica de Convergencia ", padding=10)
        self.frame_derecho.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = None

    def ejecutar_ag(self):
        # Actualizar parámetros internamente desde las entradas
        try:
            self.config["tamano_poblacion"] = int(self.entries["tamano_poblacion"].get())
            self.config["numero_generaciones"] = int(self.entries["numero_generaciones"].get())
            self.config["probabilidad_cruce"] = float(self.entries["probabilidad_cruce"].get())
            self.config["probabilidad_mutacion"] = float(self.entries["probabilidad_mutacion"].get())
        except ValueError:
            messagebox.showerror("Error", "Revisa que los parámetros sean números válidos.")
            return

        ag = AlgoritmoGenetico(self.config)
        mejor_ind, mejor_fit, historial = ag.ejecutar()

        # Mostrar en el panel de texto
        self.txt_resultados.delete("1.0", tk.END)
        self.txt_resultados.insert(tk.END, "=== ALGORITMO GENÉTICO ===\n\n")
        self.txt_resultados.insert(tk.END, f"• Mejor x: {mejor_ind:.6f}\n")
        self.txt_resultados.insert(tk.END, f"• Mejor f(x): {mejor_fit:.6f}\n")
        self.txt_resultados.insert(tk.END, f"• Generaciones: {len(historial)}\n")
        self.txt_resultados.insert(tk.END, "\n¡Ejecución completada!")

        # Mostrar gráfica incrustada
        self.dibujar_grafica(historial)

    def ejecutar_validacion(self):
        x_opt, f_opt = validacion_fuerza_bruta()
        
        self.txt_resultados.insert(tk.END, "\n\n=== FUERZA BRUTA ===\n")
        self.txt_resultados.insert(tk.END, f"• x óptimo: {x_opt:.6f}\n")
        self.txt_resultados.insert(tk.END, f"• f(x) máx: {f_opt:.6f}\n")

    def dibujar_grafica(self, historial):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.plot(historial, color='#1f77b4', linewidth=2, label='Mejor Aptitud')
        ax.set_title('Convergencia del Fitness vs Generación', fontsize=11, fontweight='bold')
        ax.set_xlabel('Generación')
        ax.set_ylabel('Aptitud f(x)')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(fig, master=self.frame_derecho)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionAG(root)
    root.mainloop()
