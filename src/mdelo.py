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
import warnings
from pathlib import Path
import random

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Descomenta esta linea si te molesta el warning de sklearn sobre feature names:
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Importa aqui los modelos que vayas a usar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# TODO: Importa los modelos que necesites (KNN, DecisionTree, RandomForest, etc.)
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier


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


# =============================================================================
# PARTE 1: EXTRACCION DE DATOS (30% de la nota)
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:

    if ruta_csv is None:
        ruta_csv = RUTA_DATOS
    try:
        df = pd.read_csv(ruta_csv)

        df.rename(columns={"Ronda": "ronda",
                           "Jugador 1": "j1_jugada",
                           "Jugador 2": "j2_jugada",
                           "Resultado": "resultado"}, inplace=True)


        return df[df['j1_jugada'].isin(JUGADA_A_NUM.keys())].reset_index(drop=True)

    except FileNotFoundError:
        print(f"[ERROR] Archivo no encontrado en: {ruta_csv}. Juega unas rondas primero.")
        return None
    except Exception as e:
        print(f"[ERROR] Error al leer el CSV: {e}")
        return None


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        return pd.DataFrame()

    df_prep = df.copy()

    df_prep['j1_num'] = df_prep['j1_jugada'].map(JUGADA_A_NUM)
    df_prep['j2_num'] = df_prep['j2_jugada'].map(JUGADA_A_NUM)

    df_prep['proxima_jugada_j1'] = df_prep['j1_num'].shift(-1)

    df_prep.dropna(inplace=True)

    df_prep['proxima_jugada_j1'] = df_prep['proxima_jugada_j1'].astype(int)

    return df_prep


# =============================================================================
# PARTE 2: FEATURE ENGINEERING (30% de la nota)
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # ------------------------------------------
    # TODO: Feature 1 - Frecuencia de jugadas
    # ------------------------------------------
    # Calcula que porcentaje de veces j2 juega cada opcion
    # Pista: usa expanding().mean() o rolling()
    df['freq_piedra'] = (df['j1_num'] == 0).expanding().mean()
    df['freq_papel'] = (df['j1_num'] == 1).expanding().mean()
    df['freq_tijera'] = (df['j1_num'] == 2).expanding().mean()

    df['resultado_prev'] = df['resultado'].shift(1).map({"Gana J1": 1, "Gana J2": -1, "Empate": 0})

    # ------------------------------------------
    # TODO: Feature 2 - Lag features (jugadas anteriores)
    # ------------------------------------------
    # Crea columnas con las ultimas 1, 2, 3 jugadas
    # Pista: usa shift(1), shift(2), etc.
    df['j1_lag_1'] = df['j1_num'].shift(1)
    df['j1_lag_2'] = df['j1_num'].shift(2)
    df['j2_lag_1'] = df['j2_num'].shift(1)

    df = df.dropna().reset_index(drop=True)

    # ------------------------------------------
    # TODO: Feature 3 - Resultado anterior
    # ------------------------------------------
    # Crea una columna con el resultado de la ronda anterior
    # Esto puede revelar patrones (ej: siempre cambia despues de perder)
    # ------------------------------------------
    # TODO: Mas features (opcional pero recomendado)
    # ------------------------------------------
    # Agrega mas features que creas utiles
    # Recuerda: mas features relevantes = mejor prediccion

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
    global FEATURES_COLS

    FEATURE_COLS = [
        'j1_lag_1',  # Jugada anterior del oponente
        'j1_lag_2',  # Jugada de hace dos rondas del oponente
        'j2_lag_1',  # Mi jugada anterior
        'freq_piedra',
        'freq_papel',
        'freq_tijera',
        'resultado_prev'
    ]

    X = df[FEATURE_COLS]

    y = df['proxima_jugada_j1']

    return X, y


# =============================================================================
# PARTE 3: ENTRENAMIENTO Y FUNCIONAMIENTO (40% de la nota)
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):

    if X.shape[0] < 5:
        print("[AVISO] Muy pocos datos para entrenar. Retornando un modelo vacío.")
        return KNeighborsClassifier()
    # TODO: Divide los datos
    # X_train, X_test, y_train, y_test = train_test_split(...)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    modelos = {
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier()
    }

    print("\n[Modelo: K-Nearest Neighbors (K=3)]")

    modelo = KNeighborsClassifier()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    score = accuracy_score(y_test, y_pred)

    print(f"Accuracy (Precisión) en Test: {score:.4f}")
    print("Informe de Clasificacion (Precisión por jugada):")
    print(classification_report(y_test, y_pred, target_names=JUGADA_A_NUM.keys(), zero_division=0))

    return modelo


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

        try:
            self.modelo = cargar_modelo(ruta_modelo)
        except FileNotFoundError:
            print("Modelo no encontrado. Entrena primero.")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):

        j1_num = JUGADA_A_NUM[jugada_j1]
        j2_num = JUGADA_A_NUM[jugada_j2]
        self.historial.append((j1_num, j2_num))

    def obtener_features_actuales(self) -> np.ndarray:
        hist = self.historial
        n = len(hist)

        j1_lag_1 = hist[-1][0] if n >= 1 else random.choice(list(JUGADA_A_NUM.values()))
        j1_lag_2 = hist[-2][0] if n >= 2 else random.choice(list(JUGADA_A_NUM.values()))
        j2_lag_1 = hist[-1][1] if n >= 1 else random.choice(list(JUGADA_A_NUM.values()))

        if n > 0:
            freq_piedra = sum(h[0] == 0 for h in hist) / n
            freq_papel = sum(h[0] == 1 for h in hist) / n
            freq_tijera = sum(h[0] == 2 for h in hist) / n
        else:
            freq_piedra = freq_papel = freq_tijera = 0

        if n >= 1:
            # Suponiendo que la ronda anterior se evalúa comparando j1 y j2
            ultimo_j1, ultimo_j2 = hist[-1]
            if ultimo_j1 == ultimo_j2:
                resultado_prev = 0
            elif (ultimo_j1 - ultimo_j2) % 3 == 1:
                resultado_prev = 1
            else:
                resultado_prev = -1
        else:
            resultado_prev = 0

        features = np.array([
            j1_lag_1, j1_lag_2, j2_lag_1,
            freq_piedra, freq_papel, freq_tijera,
            resultado_prev
        ]).reshape(1, -1)

        return features


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
        try:
            features = self.obtener_features_actuales()
            prediccion_num = self.modelo.predict(features)[0]
            return NUM_A_JUGADA[prediccion_num]
        except Exception as e:

            print(f"[ERROR PREDICCIÓN] Fallo: {e}. Jugando aleatorio.")
            return random.choice(["piedra", "papel", "tijera"])


        #Exception
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
        return
    # 2. Preparar datos
    print("\n[PASO 2] Preparando datos (Conversión a números y Target)...")
    df_prep = preparar_datos(df_raw)

    if df_prep.empty:
        print("[FIN] Datos insuficientes.")
        return
    # 3. Crear features
    print("\n[PASO 3] Creando features (pistas)...")
    df_features = crear_features(df_prep)

    if df_features.empty:
        print("[FIN] Datos insuficientes después de crear features.")
        return
    # 4. Seleccionar features
    print("\n[PASO 4] Seleccionando features (X e y)...")
    X, y = seleccionar_features(df_features)
    print(f"-> {X.shape[0]} rondas listas para entrenar.")

    if X.shape[0] < 5:
        print("[FIN] Se requieren al menos 5 rondas para un entrenamiento significativo.")
        return
    # 5. Entrenar modelo
    print("\n[PASO 5] Entrenando el modelo KNN...")
    mejor_modelo = entrenar_modelo(X, y)
    # 6. Guardar modelo
    print("\n[PASO 6] Guardando el modelo entrenado...")
    guardar_modelo(mejor_modelo, RUTA_MODELO)

    print("\n[FINALIZADO] ¡El modelo ha sido entrenado! Ahora puedes ejecutar 'evaluador.py'.")

    print("\n[!] Implementa las funciones marcadas con TODO")
    print("[!] Luego ejecuta este script para entrenar tu modelo")


if __name__ == "__main__":
    main()