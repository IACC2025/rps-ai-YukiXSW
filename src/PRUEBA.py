"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera
=================================================

INSTRUCCIONES PARA EL ALUMNO:
-----------------------------
Este archivo contiene la plantilla para tu modelo de IA.
Debes completar las secciones marcadas con TODO.

El objetivo es crear un modelo que prediga la PROXIMA jugada del oponente
y responda con la jugada que le gana.

FORMATO DEL CSV (minimo requerido):
-----------------------------------
Tu archivo data/partidas.csv debe tener AL MENOS estas columnas:
    - numero_ronda: Numero de la ronda (1, 2, 3...)
    - jugada_j1: Jugada del jugador 1 (piedra/papel/tijera)
    - jugada_j2: Jugada del jugador 2/oponente (piedra/papel/tijera)

Ejemplo:
    numero_ronda,jugada_j1,jugada_j2
    1,piedra,papel
    2,tijera,piedra
    3,papel,papel

Si has capturado datos adicionales (tiempo_reaccion, timestamp, etc.),
puedes usarlos para crear features extra.

EVALUACION:
- 30% Extraccion de datos (documentado en DATOS.md)
- 30% Feature Engineering
- 40% Entrenamiento y funcionamiento del modelo

FLUJO:
1. Cargar datos del CSV
2. Crear features (caracteristicas predictivas)
3. Entrenar modelo(s)
4. Evaluar y seleccionar el mejor
5. Usar el modelo para predecir y jugar
"""

import os
import pickle
import random
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# Descomenta esta linea si te molesta el warning de sklearn sobre feature names:
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Importa aqui los modelos que vayas a usar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# TODO: Importa los modelos que necesites (KNN, DecisionTree, RandomForest, etc.)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Configuracion de rutas
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "jugadas.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

# Mapeo de jugadas a numeros (para el modelo)
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}

# Que jugada gana a cual
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}

FEATURE_COLS = []

# =============================================================================
# PARTE 1: EXTRACCION DE DATOS (30% de la nota)
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    """
    Carga los datos del CSV de partidas.

    TODO: Implementa esta funcion
    - Usa pandas para leer el CSV
    - Maneja el caso de que el archivo no exista
    - Verifica que tenga las columnas necesarias

    Args:
        ruta_csv: Ruta al archivo CSV (usa RUTA_DATOS por defecto)

    Returns:
        DataFrame con los datos de las partidas
    """
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    try:
        columnas_requeridas = ["Ronda", "Jugador 1", "Jugador 2", "Resultado"]
        df = pd.read_csv(ruta_csv)

        # Verificar columnas
        if not all(col in df.columns for col in columnas_requeridas):
            print(f"[ERROR] El archivo CSV en {ruta_csv} no tiene las columnas requeridas: {columnas_requeridas}")
            return None

        # Renombrar para mayor claridad en el modelo
        df.rename(columns={"Ronda": "numero_ronda",
                           "Jugador 1": "jugada_j1",
                           "Jugador 2": "jugada_j2",
                           "Resultado": "resultado"}, inplace=True)

        return df[df['jugada_j1'].isin(JUGADA_A_NUM.keys())].reset_index(drop=True)

    except FileNotFoundError:
        print(f"[ERROR] Archivo no encontrado en: {ruta_csv}")
        return None
    except Exception as e:
        print(f"[ERROR] Error al leer el CSV: {e}")
        return None


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara los datos para el modelo.

    TODO: Implementa esta funcion
    - Convierte las jugadas de texto a numeros
    - Crea la columna 'proxima_jugada_j2' (el target a predecir)
    - Elimina filas con valores nulos

    Args:
        df: DataFrame con los datos crudos

    Returns:
        DataFrame preparado para feature engineering
    """
    if df is None or df.empty:
        return pd.DataFrame()
    # TODO: Implementa la preparacion de datos
    # Pistas:
    # - Usa map() con JUGADA_A_NUM para convertir jugadas a numeros
    # - Usa shift(-1) para crear la columna de proxima jugada
    # - Usa dropna() para eliminar filas con NaN
    df_prep = df.copy()

    # Convierte las jugadas de texto a numeros
    df_prep['jugada_j1_num'] = df_prep['jugada_j1'].map(JUGADA_A_NUM)
    df_prep['jugada_j2_num'] = df_prep['jugada_j2'].map(JUGADA_A_NUM)

    # Crea la columna 'proxima_jugada_j2' (el target a predecir)
    # Target es la jugada del oponente (j2) en la siguiente ronda.
    df_prep['proxima_jugada_j2'] = df_prep['jugada_j2_num'].shift(-1)

    # Crear una columna binaria del resultado (1=J1 gana, 0=Empate/J2 gana)
    # Se usara para una feature de 'resultado anterior'
    df_prep['j1_gana'] = (df_prep['resultado'] == 'Jugador 1 gana').astype(int)

    # Elimina filas con valores nulos (la ultima fila tendra NaN en 'proxima_jugada_j2')
    df_prep.dropna(inplace=True)

    # Convertir a enteros despues de dropna
    df_prep['jugada_j1_num'] = df_prep['jugada_j1_num'].astype(int)
    df_prep['jugada_j2_num'] = df_prep['jugada_j2_num'].astype(int)
    df_prep['proxima_jugada_j2'] = df_prep['proxima_jugada_j2'].astype(int)

    return df_prep



# =============================================================================
# PARTE 2: FEATURE ENGINEERING (30% de la nota)
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las features (caracteristicas) para el modelo.

    TODO: Implementa al menos 3 tipos de features diferentes.

    Ideas de features:
    1. Frecuencia de cada jugada del oponente (j2)
    2. Ultimas N jugadas (lag features)
    3. Resultado de la ronda anterior
    4. Racha actual (victorias/derrotas consecutivas)
    5. Patron despues de ganar/perder
    6. Fase del juego (inicio/medio/final)

    Cuantas mas features relevantes crees, mejor podra predecir tu modelo.

    Args:
        df: DataFrame con datos preparados

    Returns:
        DataFrame con todas las features creadas
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # ------------------------------------------
    # TODO: Feature 1 - Frecuencia de jugadas
    # ------------------------------------------
    # Calcula que porcentaje de veces j2 juega cada opcion
    # Pista: usa expanding().mean() o rolling()
    df['freq_p_j2'] = (df['jugada_j2_num'] == 0).expanding().mean().shift(1).fillna(1 / 3)
    # Frecuencia de Papel (1)
    df['freq_a_j2'] = (df['jugada_j2_num'] == 1).expanding().mean().shift(1).fillna(1 / 3)
    # Frecuencia de Tijera (2)
    df['freq_t_j2'] = (df['jugada_j2_num'] == 2).expanding().mean().shift(1).fillna(1 / 3)
    # ------------------------------------------
    # TODO: Feature 2 - Lag features (jugadas anteriores)
    # ------------------------------------------
    # Crea columnas con las ultimas 1, 2, 3 jugadas
    # Pista: usa shift(1), shift(2), etc.
    for i in range(1, 4):  # Lag 1, 2, 3
        df[f'j2_lag_{i}'] = df['jugada_j2_num'].shift(i).fillna(random.choice(list(JUGADA_A_NUM.values())))
        # Tambien incluimos la jugada de la IA (J1) para ver patrones de reaccion
        df[f'j1_lag_{i}'] = df['jugada_j1_num'].shift(i).fillna(random.choice(list(JUGADA_A_NUM.values())))
    # ------------------------------------------
    # TODO: Feature 3 - Resultado anterior
    # ------------------------------------------
    # Crea una columna con el resultado de la ronda anterior
    # Esto puede revelar patrones (ej: siempre cambia despues de perder)
    df['j1_gano_lag_1'] = df['j1_gana'].shift(1).fillna(0) # Asumimos que la primera ronda es sin resultado previo
    # ------------------------------------------
    # TODO: Mas features (opcional pero recomendado)
    # ------------------------------------------
    # Agrega mas features que creas utiles
    # Recuerda: mas features relevantes = mejor prediccion
    df['j2_rolling_mode'] = df['jugada_j2_num'].rolling(window=5).apply(lambda x: x.mode()[0], raw=False).shift(
        1).fillna(random.choice(list(JUGADA_A_NUM.values())))

    return df.dropna().reset_index(drop=True)



def seleccionar_features(df: pd.DataFrame) -> tuple:
    """
    Selecciona las features para entrenar y el target.

    TODO: Implementa esta funcion
    - Define que columnas usar como features (X)
    - Define la columna target (y) - debe ser 'proxima_jugada_j2'
    - Elimina filas con valores nulos

    Returns:
        (X, y) - Features y target como arrays/DataFrames
    """
    if df is None or df.empty:
        return np.array([]), np.array([])
    # TODO: Selecciona las columnas de features
    # feature_cols = ['feature1', 'feature2', ...]
    global FEATURE_COLS
    FEATURE_COLS = [
        # Frecuencia
        'freq_p_j2', 'freq_a_j2', 'freq_t_j2',
        # Lag de jugadas de J2
        'j2_lag_1', 'j2_lag_2', 'j2_lag_3',
        # Lag de jugadas de J1 (para ver patrones de reaccion del oponente)
        'j1_lag_1', 'j1_lag_2', 'j1_lag_3',
        # Resultado anterior
        'j1_gano_lag_1',
        # Modo de las ultimas 5 jugadas
        'j2_rolling_mode'
    ]

    df.dropna(subset=FEATURE_COLS + ['proxima_jugada_j2'], inplace=True)
    # TODO: Crea X (features) e y (target)
    # X = df[feature_cols]
    # y = df['proxima_jugada_j2']
    X = df[FEATURE_COLS]
    y = df['proxima_jugada_j2']

    return X, y



# =============================================================================
# PARTE 3: ENTRENAMIENTO Y FUNCIONAMIENTO (40% de la nota)
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    """
    Entrena el modelo de prediccion.

    TODO: Implementa esta funcion
    - Divide los datos en train/test
    - Entrena al menos 2 modelos diferentes
    - Evalua cada modelo y selecciona el mejor
    - Muestra metricas de evaluacion

    Args:
        X: Features
        y: Target (proxima jugada del oponente)
        test_size: Proporcion de datos para test

    Returns:
        El mejor modelo entrenado
    """
    if X.shape[0] == 0:
        print("[AVISO] No hay datos suficientes para entrenar. Retornando modelo aleatorio (Decision Tree).")
        return DecisionTreeClassifier()
    # TODO: Divide los datos
    # X_train, X_test, y_train, y_test = train_test_split(...)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    # TODO: Entrena varios modelos
    modelos = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier()
    }

    mejor_modelo = None
    mejor_score = -1
    mejor_nombre = ""

    # TODO: Evalua cada modelo
    # Para cada modelo:
    #   - Entrena con fit()
    #   - Predice con predict()
    #   - Calcula accuracy con accuracy_score()
    #   - Muestra classification_report()
    print("\n--- Entrenamiento y Evaluacion de Modelos ---")
    for nombre, modelo in modelos.items():
        # Entrena
        modelo.fit(X_train, y_train)

        # Predice
        y_pred = modelo.predict(X_test)

        # Evalua
        score = accuracy_score(y_test, y_pred)

        print(f"\n[Modelo: {nombre}]")
        print(f"Accuracy en Test: {score:.4f}")
        print("Informe de Clasificacion:")
        print(classification_report(y_test, y_pred, target_names=JUGADA_A_NUM.keys(), zero_division=0))
        # print("Matriz de Confusión:\n", confusion_matrix(y_test, y_pred)) # Opcional: descomentar para mas detalles

        # Selecciona el mejor
        if score > mejor_score:
            mejor_score = score
            mejor_modelo = modelo
            mejor_nombre = nombre

    # TODO: Selecciona y retorna el mejor modelo
    print(f"\n[SELECCIONADO] El mejor modelo es: {mejor_nombre} con Accuracy: {mejor_score:.4f}")
    return mejor_modelo



def guardar_modelo(modelo, ruta: str = None):
    """Guarda el modelo entrenado en un archivo."""
    if ruta is None:
        ruta = RUTA_MODELO

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"Modelo guardado en: {ruta}")


def cargar_modelo(ruta: str = None):
    """Carga un modelo previamente entrenado."""
    if ruta is None:
        ruta = RUTA_MODELO

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontro el modelo en: {ruta}")

    with open(ruta, "rb") as f:
        return pickle.load(f)


# =============================================================================
# PARTE 4: PREDICCION Y JUEGO
# =============================================================================

class JugadorIA:
    """
    Clase que encapsula el modelo para jugar.

    TODO: Completa esta clase para que pueda:
    - Cargar un modelo entrenado
    - Mantener historial de la partida actual
    - Predecir la proxima jugada del oponente
    - Decidir que jugada hacer para ganar
    """

    def __init__(self, ruta_modelo: str = None):
        """Inicializa el jugador IA."""
        self.modelo = None
        self.historial = []  # Lista de (jugada_j1, jugada_j2)

        # Columnas de features usadas en entrenamiento
        self.feature_cols = FEATURE_COLS
        # Columnas de jugadas numericas (para facilitar el feature engineering en vivo)
        self.cols_historial_num = ['jugada_j1_num', 'jugada_j2_num', 'j1sgana']

        # TODO: Carga el modelo si existe
        try:
            self.modelo = cargar_modelo(ruta_modelo)
        except FileNotFoundError:
            print("Modelo no encontrado. Entrena primero.")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        """
        Registra una ronda jugada para actualizar el historial.

        Args:
            jugada_j1: Jugada del jugador 1
            jugada_j2: Jugada del oponente
        """

        resultado = "Jugador 1 gana" if GANA_A[jugada_j1] == jugada_j2 else \
            "Empate" if jugada_j1 == jugada_j2 else \
                "Jugador 2 gana"

        j1_gana = 1 if resultado == "Jugador 1 gana" else 0

        # Registramos las jugadas convertidas a numeros y el resultado (j1_gana)
        ronda_num = [JUGADA_A_NUM[jugada_j1], JUGADA_A_NUM[jugada_j2], j1_gana]

        self.historial.append((jugada_j1, jugada_j2))

    def obtener_features_actuales(self) -> np.ndarray:
        """
        Genera las features basadas en el historial actual.

        TODO: Implementa esta funcion
        - Usa el historial para calcular las mismas features que usaste en entrenamiento
        - Retorna un array con las features

        Returns:
            Array con las features para la prediccion
        """
        if not self.historial:
            return np.array([np.nan] * len(self.feature_cols)).reshape(1,-1)
        # TODO: Calcula las features basadas en self.historial
        # Deben ser LAS MISMAS features que usaste para entrenar
        df_historial = pd.DataFrame(self.historial, columns=self.cols_historial_num)

        df_features = df_historial.copy()

        df_features['freq_p_j2'] = (df_features['jugada_j2_num'] == 0).expanding().mean().shift(1).fillna(1 / 3)
        df_features['freq_a_j2'] = (df_features['jugada_j2_num'] == 1).expanding().mean().shift(1).fillna(1 / 3)
        df_features['freq_t_j2'] = (df_features['jugada_j2_num'] == 2).expanding().mean().shift(1).fillna(1 / 3)

        for i in range(1, 4):  # Lag 1, 2, 3
            default_val = random.choice(list(JUGADA_A_NUM.values()))
            df_features[f'j2_lag_{i}'] = df_features['jugada_j2_num'].shift(i).fillna(default_val)
            df_features[f'j1_lag_{i}'] = df_features['jugada_j1_num'].shift(i).fillna(default_val)

        df_features['j1_gano_lag_1'] = df_features['j1_gana'].shift(1).fillna(0)

        default_mode = random.choice(list(JUGADA_A_NUM.values()))
        df_features['j2_rolling_mode'] = df_features['jugada_j2_num'].rolling(window=5).apply(lambda x: x.mode()[0], raw=False).shift(1).fillna(default_mode)

        current_features_df = df_features.tail(1)[self.feature_cols].copy()

        default_values = {col: current_features_df[col].iloc[0] if not pd.isna(current_features_df[col].iloc[0]) else (
            1 / 3 if 'freq' in col else 0) for col in self.feature_cols}
        current_features_df = current_features_df.fillna(default_values)

        return current_features_df.values[0].reshape(1, -1)


    def predecir_jugada_oponente(self) -> str:
        """
        Predice la proxima jugada del oponente.

        TODO: Implementa esta funcion
        - Usa obtener_features_actuales() para obtener las features
        - Usa el modelo para predecir
        - Convierte la prediccion numerica a texto

        Returns:
            Jugada predicha del oponente (piedra/papel/tijera)
        """


        if self.modelo is None:
            # Si no hay modelo, juega aleatorio
            return np.random.choice(["piedra", "papel", "tijera"])


        # TODO: Implementa la prediccion
        # features = self.obtener_features_actuales()
        # prediccion = self.modelo.predict([features])[0]
        # return NUM_A_JUGADA[prediccion]

        try:
            features = self.obtener_features_actuales()

            # features es un array 2D: [[feature1, feature2, ...]]
            prediccion_num = self.modelo.predict(features)[0]

            return NUM_A_JUGADA[prediccion_num]
        except Exception as e:
            # En caso de error, volvemos al comportamiento aleatorio
            print(f"[ERROR PREDICCION] Fallo al predecir: {e}. Jugando aleatorio.")
            return random.choice(["piedra", "papel", "tijera"])

    def decidir_jugada(self) -> str:
        """
        Decide que jugada hacer para ganar al oponente.

        Returns:
            La jugada que gana a la prediccion del oponente
        """
        prediccion_oponente = self.predecir_jugada_oponente()

        if prediccion_oponente is None:
            return np.random.choice(["piedra", "papel", "tijera"])

        # Juega lo que le gana a la prediccion
        return PIERDE_CONTRA[prediccion_oponente]


# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():
    """
    Funcion principal para entrenar el modelo.

    Ejecuta: python src/modelo.py
    """
    print("="*50)
    print("   RPSAI - Entrenamiento del Modelo")
    print("="*50)

    # TODO: Implementa el flujo completo:
    # 1. Cargar datos
    print("\n[PASO 1] Cargando datos...")
    df_raw = cargar_datos()

    if df_raw is None or df_raw.empty:
        print("[FIN] No se pudo cargar el CSV. Asegurate de jugar rondas primero.")
        return

    print(f"-> {len(df_raw)} rondas cargadas.")
    # 2. Preparar datos
    print("\n[PASO 2] Preparando datos...")
    df_prep = preparar_datos(df_raw)
    print(f"-> {len(df_prep)} rondas validas para el modelo (despues de target/dropna).")

    if df_prep.empty:
        print("[FIN] Datos insuficientes despues de la preparacion.")
        return
    # 3. Crear features
    print("\n[PASO 3] Creando features...")
    df_features = crear_features(df_prep)
    print(f"-> {len(df_features)} rondas con features completas.")

    if df_features.empty:
        print("[FIN] Datos insuficientes despues de feature engineering.")
        return
    # 4. Seleccionar features
    print("\n[PASO 4] Seleccionando features...")
    X, y = seleccionar_features(df_features)
    print(f"-> Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")

    if X.shape[0] == 0:
        print("[FIN] No hay datos despues de la seleccion de features.")
        return
    # 5. Entrenar modelo
    print("\n[PASO 5] Entrenando el modelo...")
    mejor_modelo = entrenar_modelo(X, y)

    if mejor_modelo is None:
        print("[FIN] Fallo al entrenar el modelo.")
        return
    # 6. Guardar modelo
    print("\n[PASO 6] Guardando el mejor modelo...")
    guardar_modelo(mejor_modelo)
    print("\n[!] Implementa las funciones marcadas con TODO")
    print("[!] Luego ejecuta este script para entrenar tu modelo")


if __name__ == "__main__":
    main()