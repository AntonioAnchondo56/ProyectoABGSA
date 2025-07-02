import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
import pandas as pd
from scipy.stats import f_oneway

# Calcular la función a minimizar
def fx_a_minimizar(x_1, x_2, x_3, x_4):
  valor_fx = (1.10471)*(x_1**2)*(x_2) + (0.04811)*(x_3)*(x_4)*(14 + x_2)
  return valor_fx

# Tau
def 𝝉(x_1, x_2, x_3, x_4):
  eme = m(x_2)
  ere = r(x_1, x_2, x_3)
  jota = j(x_1, x_2, x_3)
  tau_p = 𝝉_p(x_1, x_2)
  tau_p_p = 𝝉_p_p(eme, ere, jota)
  valor_tau = ((tau_p)**2 + (2*tau_p*tau_p_p*((x_2)/(2*ere))) + (tau_p_p)**2)**0.5
  return valor_tau

# Tau prima
def 𝝉_p(x_1, x_2):
  valor_tau_p = 6000/((2**0.5)*(x_1*x_2))
  return valor_tau_p

# Tau doble prima
def 𝝉_p_p(M, R, J):
  valor_tau_p_p = (M*R)/J
  return valor_tau_p_p

# M
def m(x_2):
  valor_m = 6000*(14 + (x_2/2))
  return valor_m

# R
def r(x_1, x_2, x_3):
  valor_r = (((x_2**2)/4)+((x_1 + x_3)/2)**2)**0.5
  return valor_r

# J
def j(x_1, x_2, x_3):
  valor_j = 2*(((2**0.5)*x_1*x_2)*(((x_2**2)/12) + ((x_1 + x_3)/2)**2))
  return valor_j

# Sigma
def σ(x_3, x_4):
  valor_sigma = 504000/(x_4 * (x_3**2))
  return valor_sigma

# Delta
def 𝜹(x_3, x_4):
  valor_delta = 2.1952/((x_3**3)*x_4)
  return valor_delta

# P sub c
def pc(x_3, x_4):
  valor_pc = 64746.022*(1 - (0.0282346*x_3))*x_3*(x_4**3)
  return valor_pc

# Calcular restriciones:
def g_1(x_1, x_2, x_3, x_4):
  valor_g_1 = 𝝉(x_1, x_2, x_3, x_4) - 13000
  cumple = False
  if valor_g_1 <= 0:
    cumple = True
  return valor_g_1, cumple

def g_2(x_3, x_4):
  valor_g_2 = σ(x_3, x_4) - 30000
  cumple = False
  if valor_g_2 <= 0:
    cumple = True
  return valor_g_2, cumple

def g_3(x_1, x_4):
  valor_g_3 = x_1 - x_4
  cumple = False
  if valor_g_3 <= 0:
    cumple = True
  return valor_g_3, cumple

def g_4(x_1, x_2, x_3, x_4):
  valor_g_4 = (0.1047*x_1**2) + (0.04811*x_3*x_4*(14 + x_2)) - (5)
  cumple = False
  if valor_g_4 <= 0:
    cumple = True
  return valor_g_4, cumple

def g_5(x_1):
  valor_g_5 = 0.125 - x_1
  cumple = False
  if valor_g_5 <= 0:
    cumple = True
  return valor_g_5, cumple

def g_6(x_3, x_4):
  valor_g_6 = 𝜹(x_3, x_4) - 0.25
  cumple = False
  if valor_g_6 <= 0:
    cumple = True
  return valor_g_6, cumple

def g_7(x_3, x_4):
  valor_g_7 = 6000 - pc(x_3, x_4)
  cumple = False
  if valor_g_7 <= 0:
    cumple = True
  return valor_g_7, cumple

# Función que verifica si se cumplen todas las condiciones (devuelve True si es el caso, False si no)
def verificar_restricciones(x_1, x_2, x_3, x_4):
  cumple_restricciones = [
  g_1(x_1, x_2, x_3, x_4),
  g_2(x_3, x_4),
  g_3(x_1, x_4),
  g_4(x_1, x_2, x_3, x_4),
  g_5(x_1),
  g_6(x_3, x_4),
  g_7(x_3, x_4)
  ]

  """
  # Ver qué restricciones no cumple
  for i, (valor, cumple) in enumerate(cumple_restricciones):
    if not cumple:
        print(f"Restriccion {i+1} violada con valor {valor}")
  """

  return  cumple_restricciones

# -----------------------------------------------------------------------------
# Parámetros del algoritmo GSA
# -----------------------------------------------------------------------------
# n_particulas      : Número de agentes (soluciones) en la población
# dimensiones       : Dimensión del problema (aquí 4 variables)
# rango             : Lista de tuplas con límites [(min, max), ...] para cada variable
# n_iteraciones     : Número de iteraciones
# G_0               : Constante gravitacional inicial
# alpha             : Parámetro de decaimiento de la constante G

# -----------------------------------------------------------------------------
# Especificaciones de cada agente:
# -----------------------------------------------------------------------------
# x_posiciones        : Posición. Es un vector de n dimensiones por agente
# v_velocidades       : Velocidad.
# masa_intermedia     : Masa intermedia.  Ma=Mp=Mi=M;
# masa_normalizada    : Masa normalizada. Suma 1
# a_aceleración       : Aceleración xd
# -----------------------------------------------------------------------------

# Función para contar soluciones factibles:
def contar_soluciones_factibles(x_posiciones):
    contador = 0
    for x in x_posiciones:
        if penalizacion(*x) == 0:
            contador += 1
    return contador

# Función para graficar a la población
def graficar_poblacion(x_posiciones, modo='factibilidad'):
    x = x_posiciones[:, 0]  # x1
    y = x_posiciones[:, 1]  # x2
    plt.figure(figsize=(6, 5))

    if modo == 'factibilidad':
        # máscara: True = factible, False = no factible
        mask = np.array([penalizacion(*xi) == 0 for xi in x_posiciones])
        plt.scatter(x[mask],   y[mask],   c='green', label='Factibles', alpha=0.7)
        plt.scatter(x[~mask],  y[~mask],  c='red',   label='No factibles', alpha=0.7)

    elif modo == 'fitness':
        # colorea según valor de fitness
        fit = evaluar_poblacion(x_posiciones)
        sc = plt.scatter(x, y, c=fit, cmap='viridis', s=50, alpha=0.8)
        plt.colorbar(sc, label='Fitness')

    else:
        plt.scatter(x, y, alpha=0.7, label='Población')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Distribución inicial de la población (x1 vs x2)')
    plt.grid(True)
    plt.legend()
    plt.show()

# Función para inicializar la población xd
def inicializar_poblacion(n_particulas, dimensiones, rango):
  # Definimos tantos arreglos como particulas queramos, con n dimensiones (4) y se inicializan con 0's
  x_posiciones = np.zeros((n_particulas, dimensiones))
  v_velocidades = np.zeros((n_particulas, dimensiones))
  # Para cada dimensión de cada particula, generamos un numero aleatorio dentro del rango permitido para las variables
  for i in range(n_particulas):
    for d in range(dimensiones):
      # Existen tantos rangos como variables, val minimo: rango[d][0], val maximo: rango[d][1]
      x_posiciones[i, d] = random.uniform(rango[d][0], rango[d][1])
      # La velocidad se deja inicializada en 0, puesto que al no haberse movido, no tienen velocidad

  # Contamos cuántos cumplen
  num_factibles = contar_soluciones_factibles(x_posiciones)
  print(f"\n ✅ {num_factibles}/{n_particulas} individuos cumplen restricciones \n")
  # Graficamos en 2D
  graficar_poblacion(x_posiciones, modo = "factibilidad")

  return x_posiciones, v_velocidades

# Función que penaliza considerando qué condiciones no cumple y por cuánto no cumple:
def penalizacion(x_1, x_2, x_3, x_4):
  penalizar = 0.0
  for valor, cumple in verificar_restricciones(x_1, x_2, x_3, x_4):
    if not cumple:
      peso_restriccion = 100
      penalizar += peso_restriccion*(abs(valor))
  return penalizar

# Función para evaluar el fitness de una partículas
def evaluar_fitness(x_1, x_2, x_3, x_4):
  # Sumamos la penalización para que al valer más, sean menos óptimas
  fitness = fx_a_minimizar(x_1, x_2, x_3, x_4) + penalizacion(x_1, x_2, x_3, x_4)
  return fitness

# Función para evaluar cada individuo de la población (solo depende de la posición)
def evaluar_poblacion(x_posiciones):
  # Desempaquetamos cada fila de x_posiciones en los argumentos de evaluar_fitness
    fitness = np.array([evaluar_fitness(*x) for x in x_posiciones])
    return fitness

# Función que asigna la masa a cada partícula o agente, Ma=Mp=Mi=M
def calcular_masas(fitness):
  worst_fitness = fitness.max() # Recordemos que a mayor fitnes, peor solución es,
  best_fitness = fitness.min()  # puesto que queremos minimizar
  # Verificamos que la mejor y peor fitness no sean iguales
  if best_fitness == worst_fitness:
    # Colocamos 1 a todas las varables de cada partículo
    masa_intermedia = np.ones_like(fitness)
  else:
    # Calculamos una masa "bruta" o intermedia para mapear el fitness
    # Como fitness es un arreglo, cada operación se va realizando en cada elemento, por tanto, masa_intermedia es un arreglo
    masa_intermedia = (fitness - worst_fitness) / (best_fitness - worst_fitness + 1e-12) # 1e-12 evita 0/0
  # Calculamos la masa normalizada
  masa_normzalizada = masa_intermedia / (masa_intermedia.sum() + 1e-12)
  return masa_normzalizada

# Calcula fuerza gravitacional resultante sobre cada agente.
def calcular_fuerza(constante_gravitacional, x_posiciones, masa_normalizada):
  # Obtenemos la forma de x_posiciones (o sea que saber cuántas filas y columnas xd)
  n_particulas, dimensiones = x_posiciones.shape # Descomponemos la tupla de x_posiciones
  F = np.zeros((n_particulas, dimensiones)) # Almacenar la fuerza aplicada en cada direccion de cada particula
  # Ciclo for para tomar 1 agente iterar en todos los demás
  for i in range(n_particulas):
    for j in range(n_particulas):
      if i != j: # Una partícula no se puede atraer a sí misma
        # Distancia euclidiana + epsilon
        distancia = np.linalg.norm(x_posiciones[i]-x_posiciones[j]) + 1e-12
        random_estocastico = random.random() #Para añadir la característica estocástica (0 - 1)
        # Ciclo for para calcular la fuerza en cada dirección
        for d in range(dimensiones):
          F[i, d] += (random_estocastico * constante_gravitacional * masa_normalizada[i] * masa_normalizada[j] * (x_posiciones[j, d] - x_posiciones[i, d])) / distancia
  return F

# Función para actualizar la posición y velocidad de las partículas
def actualizar_posicion_velocidad(v_velocidades, x_posiciones, F, masa_normalizada, rango):
  # Obtenemos la forma de v_velocidades (o sea que saber cuántas filas y columnas xd)
  n_particulas, dimensiones = v_velocidades.shape # Descomponemos la tupla de v_velocidades
  # Ciclo for para actualizar la velocidad y posición en cada dimensión
  for i in range(n_particulas):
    for d in range(dimensiones):
      v_velocidades[i, d] = random.random() * v_velocidades[i, d] + F[i, d]/(masa_normalizada[i] + 1e-12) # 1e-12 evita 0/0
      x_posiciones[i, d] = x_posiciones[i, d] + v_velocidades[i, d]
      # Verificamos que se encuentren dentro de los límites
      x_posiciones[i, d] = np.clip(x_posiciones[i, d], rango[d][0], rango[d][1]) # Clip hace que si se excede, se defina como el max o min
  return x_posiciones, v_velocidades

# Función principal del GSA
def gsa(n_particulas, n_iteraciones, G_0, alpha, dimensiones, rango):

  # 1. Inicializar la población:
  x_posiciones, v_velocidades = inicializar_poblacion(n_particulas, dimensiones, rango)
  mejor_valor = float("inf") # Inicializamos con el valor más grande posible, de modo que cualquier evaluación real sea menor
  mejor_solucion =  None # Inicializa sin valor porque aún no hay mejor solución evaluada
  
  # Guardar estadísticas del fitness para después graficarlas
  historial = {'mejor': [], 'peor': [], 'promedio': []}
  
  # Definimos parámetros de early stopping basándonos en el % de iteraciones sin mejora
  paciencia = int(n_iteraciones * 0.30)
  cuenta_sin_mejora = 0

  # Comenzamos con las n_iteraciones:
  for t in range(n_iteraciones):
    # 2. Calcular la aptitud de cada partícula de la población:
    fitness = evaluar_poblacion(x_posiciones)

    # print(f"\n Fitness: {fitness}")

    # Guardamos la aptitud mejor, peor y promedio
    historial['mejor'].append(mejor_valor)
    historial['peor'].append(fitness.max())
    historial['promedio'].append(fitness.mean())

    print(f"\nHistorial: Mejor: {historial['mejor']}")
    print(f"\nHistorial: Peor: {historial['peor']}")
    print(f"\nHistorial: Promedio: {historial['promedio']}")

    # Por defecto consideramos que no mejora
    mejora_en_esta_iter = False
                
   # Actualizar la mejor solución factible
    for i, x in enumerate(x_posiciones):
            if penalizacion(*x) == 0 and fitness[i] < mejor_valor: # Validar que los individuos solución sí cumplan las restricciones
                mejor_valor = fitness[i]
                mejor_solucion = x.copy()
                mejora_en_esta_iter = True
    # Aumentar el contador de no mejora
    if mejora_en_esta_iter:
        cuenta_sin_mejora = 0
    else:
        cuenta_sin_mejora += 1
    # Early stopping por falta de mejora
    if cuenta_sin_mejora >= paciencia:
        print(f"\nNo hubo mejora en {cuenta_sin_mejora} iteraciones, reseteando población...")
        mejor_guardada = mejor_solucion.copy()
        x_posiciones, v_velocidades = inicializar_poblacion(n_particulas, dimensiones, rango)
        x_posiciones[0] = mejor_guardada
        v_velocidades[0] = np.zeros(dimensiones)
        cuenta_sin_mejora = 0
        continue
    # 3 - 4. Actualizar masa, G, best, worst de la población:
    masa_normalizada = calcular_masas(fitness)
    G = G_0 * np.exp((-alpha * t) / n_iteraciones) # Función más común para actualizar G
    
    # 5. Calcular la fuerza, y actualizar la velocidad y posicion:
    F = calcular_fuerza(G, x_posiciones, masa_normalizada)
    x_posiciones, v_velocidades = actualizar_posicion_velocidad(v_velocidades, x_posiciones, F, masa_normalizada, rango)
    # Graficamos en 2D
    #graficar_poblacion(x_posiciones, modo = "factibilidad")
    print(f"Gen {t}: mejor = {mejor_valor:.6f} | mejor_solucion = {mejor_solucion}")
  # Una vez terminadas las iteraciones, graficamos las aptitudes
  graficar_convergencia_separada(historial)
    # Estadísticas finales de la población
  fitness_final = evaluar_poblacion(x_posiciones)
  return mejor_solucion, mejor_valor

# Define la función principal para ejecutar experimentos de GSA y luego un ANOVA sobre los resultados
def ejecutar_experimentos_y_anova(funcion_gsa, grilla_params, rango, semillas=10):
    # Inicializa la lista donde se guardarán los resultados de cada ejecución
    filas = []
    # Variables para guardar el mejor global de todas las iteraciones
    mejor_global = {
        'valor': float('inf'), # Guarda el valor objetivo mínimo
        'solucion': None,      # Almacena la solución correspondiente
        'parametros': None,    # Registra los parámetros usados
        'semilla': None,       # Registra la semilla que produjo este valor
        'iteracion': None      # (Opcional) iteración asociada
    }
    
    # Recorre todas las combinaciones posibles de parámetros (producto cartesiano)
    for combo in itertools.product(*grilla_params.values()):
        # Crea un diccionario de parámetros a partir de la tupla actual
        params = dict(zip(grilla_params.keys(), combo))
        # Marca la posición inicial de los resultados para esta combinación
        start_idx = len(filas)
        # Ejecuta el experimento con diferentes semillas para medir variabilidad
        for sem in range(semillas):
            print(f"\nProbando configuración: {params} (semilla={sem})")
            # Asegura reproducibilidad fijando las semillas de random y numpy
            random.seed(sem)
            np.random.seed(sem)
            # Llama a la función GSA con los parámetros actuales y obtiene solución y fitness
            solucion, valor = funcion_gsa(
                params['n_particulas'],
                params['n_iteraciones'],
                params['G_0'],
                params['alpha'],
                len(rango),
                rango
            )
            # Agrega un registro de esta corrida a la lista de filas
            filas.append({**params, 'semilla': sem, 'mejor_valor': valor})
            # Actualizar a la mejor solución global
            if valor < mejor_global['valor']:
                mejor_global.update({
                    'valor': valor,
                    'solucion': solucion,
                    'parametros': params.copy(),
                    'semilla': sem,
                    'iteracion': 'N/A'
                })
        # Al terminar las semillas para esta combinación, crea un DataFrame parcial
        df_combo = pd.DataFrame(filas[start_idx:])
        # Calcula estadísticas básicas de fitness: mínimo, máximo, media y desviación
        estadisticas = df_combo['mejor_valor'].agg(
            mejor='min',
            peor='max',
            promedio='mean',
            desviacion_estandar='std'
        )
        # Muestra por consola un resumen de los resultados para esta configuración
        print(f"\nResumen de fitness para {params}:")
        print(estadisticas.round(6))

    # Tras probar todas las combinaciones, convierte la lista completa en un DataFrame
    df = pd.DataFrame(filas)
    # Prepara un diccionario para guardar los resultados del ANOVA por cada parámetro
    resultados = {}
    # Para cada parámetro, agrupa los valores de fitness y aplica ANOVA de un factor
    for nombre in grilla_params:
        grupos = [g['mejor_valor'].values for _, g in df.groupby(nombre)]
        f_stat, p_val = f_oneway(*grupos)
        resultados[nombre] = {'F-statistic': f_stat, 'p-value': p_val}
    # Convierte los resultados del ANOVA en un DataFrame para una visión estructurada
    tabla_anova = pd.DataFrame.from_dict(resultados, orient='index')
    # Retorna el DataFrame completo de experimentos, la tabla de ANOVA y la mejor solución global
    return df, tabla_anova, mejor_global

def graficar_convergencia_separada(historial):
    generaciones = list(range(len(historial['mejor'])))

    # Gráfica 1: Mejor aptitud
    plt.figure(figsize=(6, 5))
    plt.plot(generaciones, historial['mejor'], color='green', label='Mejor aptitud')
    plt.ylabel('Mejor')
    plt.title('Evolución de la mejor aptitud')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Gráfica 2: Peor aptitud
    plt.figure(figsize=(6, 5))
    plt.plot(generaciones, historial['peor'], color='red', label='Peor aptitud')
    plt.ylabel('Peor')
    plt.title('Evolución de la peor aptitud')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Gráfica 3: Promedio de aptitudes
    plt.figure(figsize=(6, 5))
    plt.plot(generaciones, historial['promedio'], color='blue', label='Aptitud promedio')
    plt.xlabel('Iteración')
    plt.ylabel('Promedio')
    plt.title('Evolución de la aptitud promedio')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    # Parámetros base del GSA
    parametros = {
        'n_particulas': 150,
        'n_iteraciones': 100,
        'G_0': 50,
        'alpha': 5,
        'dimensiones': 4,
        # 'rango': [(0.125, 5), (0.1, 10), (0.1, 10), (0.125, 5)] # rangos buenos
        #'rango': [(0.125, 10), (0.1, 15), (0.1, 15), (0.125, 10)],
        'rango': [(0.1, 2), (0.1, 10), (0.1, 10), (0.1, 2)],
    }

    # Ejecución del algoritmo GSA
    solucion, valor = gsa(
        parametros['n_particulas'],
        parametros['n_iteraciones'],
        parametros['G_0'],
        parametros['alpha'],
        parametros['dimensiones'],
        parametros['rango']
    )

    print("\nMejor solución encontrada:", solucion)
    print("Valor objetivo:", valor)
    try:
      print("Restricciones cumplidas:", all(cumple for _, cumple in verificar_restricciones(*solucion)))
    except Exception as e:
      print("La solución no se pudo evaluar porque ninguna cumplió las restricciones")

    
    # Experimentos y ANOVA
    print("\nEjecutando múltiples configuraciones para ANOVA...")
    grilla = {
    'n_particulas': [50, 150, 500],
    'n_iteraciones': [40, 50, 100],
    'G_0': [50, 100],
    'alpha': [5, 10]
    }

    df_resultados, tabla_anova, mejor_global = ejecutar_experimentos_y_anova(
        gsa,
        grilla,
        parametros['rango'],
        semillas=5
    )

    print("\nMejor solución global encontrada en todos los experimentos:")
    print(f"Valor: {mejor_global['valor']}")
    print(f"Solución: {mejor_global['solucion']}")
    print(f"Parámetros: {mejor_global['parametros']}")
    print(f"Semilla: {mejor_global['semilla']}")

    print("\nTabla de resultados ANOVA:")
    print(tabla_anova)
