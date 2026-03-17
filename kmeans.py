import pandas as pd          #Para manejar tablas de datos
import numpy as np           #Para operaciones matemáticas rápidas y cálculos de distancias
import random                #Para elegir los centroides iniciales al azar 
import os                    #Para verificar si el archivo existe antes de leerlo
import matplotlib.pyplot as plt #Para dibujar la gráfica final de los cluster
from sklearn.decomposition import PCA #Visualizar la gráfica

print("="*70)
print("PRACTICA: K-MEANS CON HEOM")
print("="*70)

# ----------------CARGA DE DATOS ----------------
archivo = 'breast-cancer.data'

if not os.path.exists(archivo):
    print(f"ERROR: No encuentro '{archivo}'")
    exit()

nombres = [
    'clase', 'age', 'menopause', 'tumor-size', 'inv-nodes', 
    'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat'
]

print(f"\n[1] Cargando datos...")
try:
    df = pd.read_csv(archivo, names=nombres, skipinitialspace=True)
    
    # CAMBIO REALIZADO: Se reemplaza '?' por NaN pero NO se eliminan filas (.dropna())
    # porque HEOM maneja estos valores automáticamente asignando 1 
    df = df.replace('?', np.nan).reset_index(drop=True)
    
    print(f"    -> Datos cargados: {len(df)} registros (incluyendo valores faltantes manejados por HEOM).")
except Exception as e:
    print(f"Error al leer: {e}")
    exit()

#Separa la etiqueta real (clase) de los atributos (X)
#La etiqueta 'y_real' NO se usa, solo sirve para evaluar al final
y_real = df['clase']
X = df.drop('clase', axis=1)

#Qué columnas son numéricas y cuáles categóricas
cols_num = ['deg-malig']
cols_cat = [c for c in X.columns if c not in cols_num]

#Verifica que las columnas de texto estén limpias (sin espacios extra)
# Si el valor es NaN, lo dejamos así; si es string, quitamos espacios.
for col in X.columns:
    X[col] = X[col].apply(lambda x: str(x).strip() if pd.notna(x) else x)


#Verificar que la columna numérica realmente tiene números
X['deg-malig'] = pd.to_numeric(X['deg-malig'], errors='coerce')

# ----------------FÓRMULAS HEOM ----------------
def range_a(data):
    #Calcula el rango = max - min (usando nanmax/nanmin para ignorar huecos en el cálculo del rango global)
    r = np.nanmax(data) - np.nanmin(data)
    return r if r != 0 else 1.0

rangos = {col: range_a(X[col]) for col in cols_num}

def overlap(x, y):
    #Distancia para atributos nominales. 0 si son iguales, 1 si son diferentes
    return 0.0 if x == y else 1.0

def rn_diff(x, y, r):
    #Distancia  para atributos numéricos |x-y|/rango
    return abs(x - y) / r if r != 0 else 0.0

def d_a(x, y, col, es_num, r=None):
    #Función general que decide qué fórmula usar según el tipo de atributo
    if es_num:
        return rn_diff(x, y, r)
    else:
        return overlap(x, y)

def calcular_heom(idx1, idx2, d_num, d_cat, rangos_dict, cols_c):
    #HEOM(x,y) = sqrt( suma (d_a)^2 )
    #Calcula la distancia total entre dos pacientes sumando las diferencias de todos sus atributos.
    #REGLA HEOM: Si un atributo es desconocido (NaN), su distancia es 1 (máxima).
    
    suma_cuadrados = 0.0
    #Parte numérica
    for col in rangos_dict:
        val1 = d_num[col][idx1]
        val2 = d_num[col][idx2]
        
        # Verificación explícita de valores faltantes para aplicar la regla de distancia=1
        if np.isnan(val1) or np.isnan(val2):
            d = 1.0
        else:
            d = d_a(val1, val2, col, True, rangos_dict[col])
            
        suma_cuadrados += d ** 2

    #Parte categórica   
    for col in cols_c:
        val1 = d_cat[col][idx1]
        val2 = d_cat[col][idx2]
        
        #Verificación de valores faltantes (NaN o string 'nan')
        if pd.isna(val1) or pd.isna(val2) or str(val1) == 'nan' or str(val2) == 'nan':
            d = 1.0
        else:
            d = d_a(val1, val2, col, False)
            
        suma_cuadrados += d ** 2
        
    return np.sqrt(suma_cuadrados)

# ---------------- ALGORITMO K-MEANS ----------------
def kmeans_heom(X_df, k=2, max_iter=50):
    
    #Implementación del algoritmo K-Means usando HEOM:
    #1. Calcular distancia de cada paciente a los centroides actuales.
    #2. Generar los grupos asignando cada paciente al centroide más cercano.
    #3. Calcular nuevos centroides (primero teórico, luego buscando el objeto real más cercano).
    #4. Repetir hasta que los grupos no cambien (convergencia).
   
    n = len(X_df)
    #Prepara diccionarios con los valores en arrays para que el cálculo sea rápido
    d_num = {col: X_df[col].values for col in cols_num}
    d_cat = {col: X_df[col].values for col in cols_cat}
    
    #Inicialización aleatoria de los centroides
    centroides_idx = random.sample(range(n), k)
    
    historial = []
    errores = []
    
    for i in range(max_iter):
        etiquetas = np.zeros(n, dtype=int)
        matriz_dist = np.zeros((n, k))
        error_actual = 0.0
        
        #SACAR DISTANCIA DE CADA PACIENTE A LOS CENTROIDES
        for p in range(n):
            dists = []
            for c in range(k):
                #Calculamos la distancia HEOM entre el paciente 'p' y el centroide 'c'
                d = calcular_heom(p, centroides_idx[c], d_num, d_cat, rangos, cols_cat)
                dists.append(d)
                matriz_dist[p, c] = d

            #GENERA LOS GRUPOS
            #Asignamos el paciente al grupo del centroide que tenga la MENOR distancia
            etiquetas[p] = np.argmin(dists)
            
            #Suma de distancias al cuadrado.
            #Esto nos dice qué tan buenos son los grupos entre menor error = mejores grupos
            error_actual += min(dists) ** 2 
        
        errores.append(error_actual)
        tamanos = [int(np.sum(etiquetas == c)) for c in range(k)]
        #Guarda los resultado de la iteración para poder analizarlo después
        historial.append({
            'iter': i+1, 
            'tamanos': tamanos, 
            'matriz': matriz_dist, 
            'etiquetas': etiquetas.copy(),
            'centroides_idx': list(centroides_idx)
        })
        
        if i == 0 or (i+1) % 5 == 0:
            print(f"   Iter {i+1}: Error={error_actual:.1f}", end='\r')
        
        # CALCULAR NUEVOS CENTROIDES
        nuevos_idx = []
        for c in range(k):
            mask = (etiquetas == c)
            ids_cluster = np.where(mask)[0]
            
            if len(ids_cluster) == 0:
                nuevos_idx.append(random.randint(0, n-1))
                continue
            
            #1. Calcula el centroide "TEÓRICO" (Promedio de valores)
            # Para numéricos: Media aritmética (usamos nanmean para ignorar NaN al promediar).
            # Para categóricos: Moda (el valor que más se repite, ignorando NaN).
            teorico = {}
            for col in cols_num:
                teorico[col] = np.nanmean(d_num[col][ids_cluster])
            for col in cols_cat:
                # Filtramos valores válidos (no NaN) para calcular la moda correctamente
                validos = [v for v in d_cat[col][ids_cluster] if pd.notna(v) and str(v) != 'nan']
                if len(validos) > 0:
                    u, cnt = np.unique(validos, return_counts=True)
                    teorico[col] = u[np.argmax(cnt)]
                else:
                    teorico[col] = 'desconocido'
            
            #2. Busca el objeto REAL del dataset que esté más cerca de ese teórico (Medoide)
            # Esto es necesario porque el "promedio" de texto no existe, necesitamos un paciente real.
            mejor_d = float('inf')
            mejor_i = ids_cluster[0]
            
            for idx in ids_cluster:
                #Calculamos distancia de este paciente real al centroide teórico
                d_sq = 0.0
                
                # Parte numérica
                val_real = d_num['deg-malig'][idx]
                val_teo = teorico['deg-malig']
                if np.isnan(val_real):
                    d_sq += 1.0 ** 2
                else:
                    d_sq += ((val_real - val_teo)/rangos['deg-malig'])**2
                
                # Parte categórica
                for col in cols_cat:
                    v_real = d_cat[col][idx]
                    v_teo = teorico[col]
                    if pd.isna(v_real) or str(v_real) == 'nan':
                        d_sq += 1.0 ** 2
                    else:
                        if v_real != v_teo:
                            d_sq += 1.0 ** 2
                            
                d_tot = np.sqrt(d_sq)
                
                if d_tot < mejor_d:
                    mejor_d = d_tot
                    mejor_i = idx
            
            #Este 'mejor_i' es el NUEVO CENTROIDE para la siguiente iteración
            nuevos_idx.append(mejor_i)
        
        #PASO D: VERIFICAR CONVERGENCIA

        #Si los centroides no cambiaron de posición, ya terminamos esta ejecución.
        if sorted(centroides_idx) == sorted(nuevos_idx):
            print(f"   Convergió en iter {i+1}   ")
            break
        centroides_idx = nuevos_idx
    
    return etiquetas, centroides_idx, historial, errores

# ---------------- PASO 4: EVALUACIÓN ----------------
def evaluar(y_pred, y_true):
    #Compara los grupos obtenidos con las clases reales para calcular métricas
    #Aseguramos alineación de índices tras posible limpieza previa
    y_true_clean = y_true.reset_index(drop=True)
    y_pred_clean = pd.Series(y_pred).reset_index(drop=True)
    
    df_e = pd.DataFrame({'c': y_pred_clean, 'r': y_true_clean})
    # Filtramos filas donde la clase real sea NaN (si hubiera)
    df_e = df_e.dropna(subset=['r'])
    
    if df_e.empty:
        return {'VP':0,'FP':0,'VN':0,'FN':0,'Sensibilidad':0,'Exactitud':0,'Precision':0,'Tasa_Error':0}

    conteo_rec = {c: sum(df_e[df_e['c']==c]['r']=='recurrence-events') for c in df_e['c'].unique()}
    total_c = {c: len(df_e[df_e['c']==c]) for c in df_e['c'].unique()}
    
    cluster_pos = max(conteo_rec, key=lambda k: conteo_rec[k]/total_c[k])
    cluster_neg = 1 - cluster_pos
    
    mapeo = {cluster_pos: 'recurrence-events', cluster_neg: 'no-recurrence-events'}
    y_aj = df_e['c'].map(mapeo)
    
    bin_r = (df_e['r'] == 'recurrence-events').astype(int)
    bin_p = (y_aj == 'recurrence-events').astype(int)
    
    VP = np.sum((bin_p==1)&(bin_r==1))
    FP = np.sum((bin_p==1)&(bin_r==0))
    VN = np.sum((bin_p==0)&(bin_r==0))
    FN = np.sum((bin_p==0)&(bin_r==1))
    
    tot = VP+VN+FP+FN
    if tot==0: return {'VP':0,'FP':0,'VN':0,'FN':0,'Sensibilidad':0,'Exactitud':0,'Precision':0,'Tasa_Error':0}
    
    return {
        'VP':int(VP), 'FP':int(FP), 'VN':int(VN), 'FN':int(FN),
        'Sensibilidad': VP/(VP+FN) if (VP+FN)>0 else 0,
        'Exactitud': (VP+VN)/tot,
        'Precision': VP/(VP+FP) if (VP+FP)>0 else 0,
        'Tasa_Error': (FP+FN)/tot
    }

# ----------------REPETIR 20 VECES ----------------
print("\n" + "="*70)
print("EJECUTANDO 20 ITERACIONES")
print("="*70)

resultados = []
detalles = []
asignaciones_completas = []
todos_errores = []
todas_etiquetas = []
todos_centroides = []


#REPETIR EL PROCEDIMIENTO 20 VECES

for i in range(1, 21):
    print(f"\n--- Ejecución {i}/20 ---")
    try:
        #Llamamos a la función que hace todo el procesos (kmeans_heom)
        etq, cent, hist, errs = kmeans_heom(X, k=2)
        
        todos_errores.append(errs)
        todas_etiquetas.append(hist[-1]['etiquetas'])
        todos_centroides.append(hist[-1]['centroides_idx'])
        
        mets = evaluar(etq, y_real)
        resultados.append({
            'Ejecucion': i, 'Iteraciones': len(hist),
            'VP': mets['VP'], 'FP': mets['FP'], 'VN': mets['VN'], 'FN': mets['FN'],
            'Sensibilidad': mets['Sensibilidad'], 'Exactitud': mets['Exactitud'],
            'Precision': mets['Precision'], 'Tasa_Error': mets['Tasa_Error']
        })
        
        detalles.append({
            'Ejecucion': i, 
            'Matriz_Muestra': np.array2string(hist[-1]['matriz'][:5,:], precision=2, separator=', '),
            'Tamanos': str(hist[-1]['tamanos'])
        })
        
        # Guardo dato por dato para la hoja completa
        matriz_final = hist[-1]['matriz']
        for idx_paciente in range(len(X)):
            fila_datos = {
                'Ejecucion': i,
                'Indice_Paciente': idx_paciente,
                'Cluster_Asignado': int(etq[idx_paciente]),
                'Distancia_Centroide_0': matriz_final[idx_paciente, 0],
                'Distancia_Centroide_1': matriz_final[idx_paciente, 1]
            }
            asignaciones_completas.append(fila_datos)
        
        print(f"   Resultado: Exactitud = {mets['Exactitud']:.4f}")
        
    except Exception as e:
        print(f"   Error: {e}")

# ----------------  GUARDAR EXCEL ----------------
print("\n Guardando resultados en EVALUACION.xlsx...")
try:
    if not resultados:
        raise ValueError("No hay resultados.")
        
    df_res = pd.DataFrame(resultados)
    prom = df_res.mean(numeric_only=True)
    
    fila_prom = {
        'Ejecucion': 'PROMEDIO',
        'Iteraciones': prom['Iteraciones'],
        'VP': prom['VP'], 'FP': prom['FP'], 'VN': prom['VN'], 'FN': prom['FN'],
        'Sensibilidad': prom['Sensibilidad'], 'Exactitud': prom['Exactitud'],
        'Precision': prom['Precision'], 'Tasa_Error': prom['Tasa_Error']
    }
    
    df_final = pd.concat([df_res, pd.DataFrame([fila_prom])], ignore_index=True)
    df_completo = pd.DataFrame(asignaciones_completas)
    
    with pd.ExcelWriter('EVALUACION.xlsx', engine='openpyxl') as writer:
        
        #HOJA 1: Resultados
        #Muestra de forma resumida  las 20 ejecuciones con las métricas de evaluación (VP, FP, sensibilidad, exactitud, prescisión, tasa de error)
        #y una fila final con los PROMEDIOS de todo, para ver rápidamente qué tan bien funcionó el modelo en general.
        df_final.to_excel(writer, sheet_name='Resultados', index=False)
        
        #HOJA 2: Matrices
        #muestra pequeña (primeros 5 pacientes) de las distancias calculadas a los centroides en cada ejecución.
        #para evidencia de que el cálculo HEOM se realizó correctamente y para ver cómo varían las distancias entre ejecuciones.
        pd.DataFrame(detalles).to_excel(writer, sheet_name='Matrices', index=False)
        
        
        #HOJA 3:Asignaciones_Completas
        # Tiene filas (Total pacientes x 20 ejecuciones).
        # Muestra exactamente a qué grupo (0 o 1) fue asignado CADA paciente en CADA ejecución, junto con sus distancias.
        df_completo.to_excel(writer, sheet_name='Asignaciones_Completas', index=False)
    
    print("  Archivo generado con 3 hojas.")
    print(f"   Promedio Exactitud: {prom['Exactitud']:.4f}")
    
except Exception as e:
    print(f"   ERROR al guardar: {e}")

# ---------------- MEJOR EJECUCIÓN Y RESULTADOS ----------------
if len(todos_errores) > 0:
    #Encuenta cuál de las 20 ejecuciones tuvo el menor error total
    errores_finales = [e[-1] for e in todos_errores]
    errors = np.array(errores_finales)
    best_ind = np.where(errors == errors.min())[0].tolist()[0]
    
    print(f"\nMejor Ejecución Global detectada: #{best_ind+1}")
    print(f"    Error Mínimo: {errors.min():.2f}")
    
    mejores_etq = todas_etiquetas[best_ind]
    mejores_cent = todos_centroides[best_ind]
    #Muestra en consola cómo quedaron los grupos
    print("\n    --- Composición de Grupos (Mejor Ejecución) ---")
    df_comp = pd.DataFrame({'Cluster': mejores_etq, 'Clase': y_real.reset_index(drop=True)})
    # Eliminamos solo para la visualización de composición si la clase era NaN
    df_comp = df_comp.dropna(subset=['Clase'])
    
    for c in [0, 1]:
        sub = df_comp[df_comp['Cluster']==c]['Clase']
        rec = sum(sub=='recurrence-events')
        norec = sum(sub=='no-recurrence-events')
        print(f"    Cluster {c}: {len(sub)} pacientes ({rec} Recurrencias, {norec} No Recurrencias)")
    
    #Muestra las características de los centroides
    print("\n    --- Características de los Centroides (Medoides) ---")
    for i, idx in enumerate(mejores_cent):
        print(f"    Centroide {i} (Índice original {idx}):")
        print(X.iloc[idx].to_string())
        print("-" * 30)

    #Genera la gráfica usando PCA para visualizar los grupos 
    print("\n[8] Generando gráfica de grupos...")
    
    # Para PCA llenamos temporalmente los NaN solo para poder graficar (no afecta el clustering real)
    X_vis = X.fillna(-1)
    X_encoded = pd.get_dummies(X_vis)
    
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(X_encoded)
    
    df_pca = pd.DataFrame(data=componentes, columns=['PC1', 'PC2'])
    df_pca['Cluster'] = mejores_etq
    
    plt.figure(figsize=(10, 8))
    colors_map = {0: 'green', 1: 'blue'} 
    #Dibuja los puntos de los pacientes
    for c in [0, 1]:
        mask = (df_pca['Cluster'] == c)
        plt.scatter(df_pca.loc[mask, 'PC1'], df_pca.loc[mask, 'PC2'], 
                    c=colors_map[c], alpha=0.5, s=200, edgecolors='none', label=f'Grupo {c}')
    
    #Dibuja los centroides como círculos grandes
    centroides_data = X_encoded.iloc[mejores_cent]
    centroides_pca = pca.transform(centroides_data)
    
    for i, c in enumerate([0, 1]):
        plt.scatter(centroides_pca[i, 0], centroides_pca[i, 1], 
                    c=colors_map[c], s=1500, alpha=0.9, edgecolors='black', linewidth=2, marker='o', zorder=10)

    plt.title(f'Mejor Ejecución #{best_ind+1}: Grupos Formados por K-Means', fontsize=16)
    plt.xlabel(f'Componente 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'Componente 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    print("   Gráfica generada. Cierra la ventana para finalizar.")

print("\nFin del programa.")