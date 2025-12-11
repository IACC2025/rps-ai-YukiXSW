
import os
import pickle
import warnings
from pathlib import Path

# añadidas
import random
from typing import List, Dict
import joblib

import pandas as pd
import numpy as np

# Descomenta esta linea si te molesta el warning de sklearn sobre feature names:
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Importa aqui los modelos que vayas a usar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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
PIERDE_CONTRA = {
    "piedra": "papel",
    "papel": "tijera",
    "tijera": "piedra"}

N_LAG = 3
WINDOW_SIZE = 10
# =============================================================================
# PARTE 1: EXTRACCION DE DATOS (30% de la nota)
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:

    if ruta_csv is None:
        ruta_csv = RUTA_DATOS

    try:
        df = pd.read_csv(ruta_csv)

        df = df.rename(columns={
            'Ronda': 'numero_ronda',
            'Jugador 1': 'jugada_j1',
            'Jugador 2': 'jugada_j2'
        })

        df['jugada_j1'] = df['jugada_j1'].str.lower()
        df['jugada_j2'] = df['jugada_j2'].str.lower()

        required_cols = ['numero_ronda', 'jugada_j1', 'jugada_j2']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("El CSV no tiene las columnas requeridas (Ronda, Jugador 1, Jugador 2).")

        print(f"Datos cargados con {len(df)} rondas.")
        return df
    except FileNotFoundError:
        print(
            f"Error: No se encontró el archivo de datos en {ruta_csv}. Asegúrate de que  'jugadas.csv' este en la carpeta 'data/'.")
        return pd.DataFrame()

def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df


    df['jugada_j1_num'] = df['jugada_j1'].map(JUGADA_A_NUM)
    df['jugada_j2_num'] = df['jugada_j2'].map(JUGADA_A_NUM)


    df = df.dropna(subset=['jugada_j1_num', 'jugada_j2_num'])
    df['jugada_j1_num'] = df['jugada_j1_num'].astype(int)
    df['jugada_j2_num'] = df['jugada_j2_num'].astype(int)

    df['proxima_jugada_j2'] = df['jugada_j2_num'].shift(-1)

    df = df.dropna(subset=['proxima_jugada_j2'])
    df['proxima_jugada_j2'] = df['proxima_jugada_j2'].astype(int)

    return df


# =============================================================================
# PARTE 2: FEATURE ENGINEERING (30% de la nota)
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    # 1. Lag Features (Últimas N jugadas del oponente J2)
    # Ejemplo: ¿Qué jugó el oponente en la ronda anterior?
    for i in range(1, N_LAG + 1):
        df[f'j2_lag_{i}'] = df['jugada_j2_num'].shift(i)

    # 2. Lag Features (Últimas N jugadas de mi jugador J1)
    # Ejemplo: ¿Qué jugué yo en la ronda anterior? (Patrones de respuesta del oponente)
    for i in range(1, N_LAG + 1):
        df[f'j1_lag_{i}'] = df['jugada_j1_num'].shift(i)

    # 3. Frecuencia de la última jugada de J2 (Tendencia a repetir una opción)
    # Calcula la frecuencia de cada jugada (0, 1, 2) en las últimas 10 rondas.
    for jugada_num in JUGADA_A_NUM.values():
        df[f'freq_j2_{jugada_num}'] = (
                df['jugada_j2_num'] == jugada_num
        ).rolling(window=WINDOW_SIZE, min_periods=1).mean().shift(1)

    # Eliminar las filas que ahora tienen NaN debido al shift (las primeras N rondas)
    df = df.dropna()

    return df

def seleccionar_features(df: pd.DataFrame) -> tuple:
    if df.empty:
        return pd.DataFrame(), pd.Series()

    # Se seleccionan automáticamente todas las features creadas (lag y freq)
    feature_cols = [col for col in df.columns if 'lag' in col or 'freq' in col]

    X = df[feature_cols]
    y = df['proxima_jugada_j2']

    return X, y


# =============================================================================
# PARTE 3: ENTRENAMIENTO Y FUNCIONAMIENTO (40% de la nota)
# =============================================================================

def entrenar_modelo(X, y, n_train: int = 450):
    print("\n[INFO] Iniciando el entrenamiento de modelos...")

    # Implementar el split manual solicitado (450 para entrenamiento, el resto para prueba)
    if len(X) < n_train:
        print(f"[ALERTA] Dataset size ({len(X)}) es menor que el tamaño de entrenamiento solicitado ({n_train}).")
        print("Usando un split aleatorio 90% entrenamiento / 10% prueba.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    else:
        # Usar las primeras 450 filas para entrenamiento (Split no aleatorio)
        X_train = X.iloc[:n_train]
        y_train = y.iloc[:n_train]
        X_test = X.iloc[n_train:]
        y_test = y.iloc[n_train:]
        print(f"[INFO] Dataset dividido: {len(X_train)} para entrenamiento | {len(X_test)} para prueba.")


    # Definición de modelos a probar (al menos 2 requeridos)
    modelos = {
        'DecisionTree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }

    mejor_modelo = None
    mejor_accuracy = 0
    nombre_mejor_modelo = ""

    print("\n--- Resultados de la Evaluación ---")
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nModelo: {nombre}")
        print(f"Accuracy (Test): {accuracy:.4f}")
        # print(classification_report(y_test, y_pred))

        if accuracy > mejor_accuracy:
            mejor_accuracy = accuracy
            mejor_modelo = modelo
            nombre_mejor_modelo = nombre

    print(f"\n>>> Mejor modelo seleccionado: {nombre_mejor_modelo} con Accuracy: {mejor_accuracy:.4f}")

    return mejor_modelo


def guardar_modelo(modelo, ruta: Path = RUTA_MODELO):
    """Guarda el modelo entrenado en un archivo."""
    try:
        ruta.parent.mkdir(parents=True, exist_ok=True)  # Crear carpeta 'models' si no existe
        joblib.dump(modelo, ruta)
        print(f"\n[ÉXITO] Modelo guardado en: {ruta}")
    except Exception as e:
        print(f"\n[ERROR] No se pudo guardar el modelo: {e}")

def cargar_modelo(ruta: Path = RUTA_MODELO):
    """Carga el modelo entrenado desde disco."""
    if not ruta.exists():
        raise FileNotFoundError(f"Modelo no encontrado en {ruta}")
    return joblib.load(ruta)

# =============================================================================
# PARTE 4: PREDICCION Y JUEGO
# =============================================================================

class JugadorIA:

    def __init__(self, ruta_modelo: str = None):
        """Inicializa el jugador IA."""
        self.modelo = None
        self.historial: List[Dict[str, int]] = []

        if ruta_modelo is None:
            ruta_modelo = RUTA_MODELO
        try:
            self.modelo = cargar_modelo(ruta_modelo)
            print("[INFO] Modelo IA cargado con éxito")
        except FileNotFoundError:
            print("Modelo no encontrado. Entrena primero.")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):

        self.historial.append({
            'jugada_j1_num': JUGADA_A_NUM.get(jugada_j1.lower()),
            'jugada_j2_num': JUGADA_A_NUM.get(jugada_j2.lower())
        })

    def obtener_features_actuales(self) -> np.ndarray:
        """
        Calcula el vector de features para la última ronda del historial.
        Debe replicar exactamente la lógica de crear_features.
        """
        # Convertir historial a DataFrame
        df_historial = pd.DataFrame(self.historial)

        # Necesitamos un mínimo de filas para calcular los features (el máximo es WINDOW_SIZE)
        MIN_ROWS = max(N_LAG, WINDOW_SIZE) + 1

        if len(df_historial) < MIN_ROWS:
            return None  # No hay suficientes datos para una predicción con features

        # Aplicar el Feature Engineering (solo la parte de los features)
        df_features = crear_features(df_historial)

        # Si el DataFrame de features está vacío después del dropna, es que no hay suficientes datos
        if df_features.empty:
            return None

        # Seleccionar las columnas de features
        feature_cols = [col for col in df_features.columns if 'lag' in col or 'freq' in col]

        # Devolver el vector de features de la última ronda (la que se acaba de jugar)
        return df_features.iloc[-1][feature_cols].values

    def predecir_jugada_oponente(self) -> str:
        """
        Utiliza el modelo entrenado para predecir la próxima jugada del oponente.
        """
        if self.modelo is None:
            # Si no hay modelo, juega aleatorio
            return random.choice(["piedra", "papel", "tijera"])

        try:
            features = self.obtener_features_actuales()

            if features is None:
                # Si el historial es corto, juega aleatorio
                return random.choice(["piedra", "papel", "tijera"])

            # La predicción es un número (0, 1, o 2)
            # El modelo.predict espera un array 2D
            prediccion_num = self.modelo.predict([features])[0]

            # Convertir el número predicho de vuelta a string
            return NUM_A_JUGADA[prediccion_num]

        except Exception as e:
            # En caso de error, juega aleatorio
            # print(f"Error en predicción: {e}. Jugando aleatorio.")
            return random.choice(["piedra", "papel", "tijera"])

    def decidir_jugada(self) -> str:
        """
        Decide qué jugada hacer para ganar al oponente.
        """
        # 1. Predecir lo que jugará el oponente
        prediccion_oponente = self.predecir_jugada_oponente()

        if prediccion_oponente is None:
            return random.choice(["piedra", "papel", "tijera"])

        # 2. Jugar lo que le gana a la predicción
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


    df = cargar_datos()
    if df.empty:
        return

    df_preparado = preparar_datos(df)

    df_features = crear_features(df_preparado)
    print(f"[INFO] Total de muestras utilizables (con features): {len(df_features)}")

    X, y = seleccionar_features(df_features)
    if X.empty:
        print(
            "[ERROR] El Feature Engineering resultó en un conjunto de features vacío. Asegúrate de tener suficientes datos.")
        return

    mejor_modelo = entrenar_modelo(X, y, n_train=450)

    if mejor_modelo:
        guardar_modelo(mejor_modelo)

    print("\n--- Entrenamiento Finalizado. Ejecuta 'python src/evaluador.py' para probar la IA. ---")
    # print("[!] Luego ejecuta este script para entrenar tu modelo")


if __name__ == "__main__":
    main()
