import time
import keyboard
import csv
import os

# Mapeo de teclas y opciones
opciones_j1 = {"a": "piedra", "s": "papel", "d": "tijera"}
opciones_j2 = {"1": "piedra", "2": "papel", "3": "tijera"}

# Archivo CSV
archivo_csv = "jugadas.csv"


def inicializar_csv():
    """Crea el archivo CSV si no existe."""
    if not os.path.exists(archivo_csv):
        with open(archivo_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Ronda", "Jugador 1", "Jugador 2", "Resultado"])


def cuenta_regresiva():
    """Cuenta regresiva personalizada para piedra, papel o tijera."""
    pasos = ["piedra", "papel", "tijera", "1", "2", "3"]

    print("Preparados...")
    for paso in pasos:
        print(paso)
        time.sleep(1)
    print("¡YA!\n")


def leer_jugada(opciones):
    """Lee la jugada de un jugador hasta que pulse una tecla válida."""
    while True:
        for tecla, jugada in opciones.items():
            if keyboard.is_pressed(tecla):
                return jugada

        # Salida con Q
        if keyboard.is_pressed("q"):
            print("Un jugador ha salido del juego.")
            exit()


def determinar_ganador(j1, j2):
    """Devuelve el resultado de la ronda."""
    if j1 == j2:
        return "Empate"

    reglas = {
        ("piedra", "tijera"),
        ("papel", "piedra"),
        ("tijera", "papel")
    }

    return "Jugador 1 gana" if (j1, j2) in reglas else "Jugador 2 gana"


def guardar_csv(ronda, j1, j2, resultado):
    """Guarda cada ronda en el archivo CSV."""
    with open(archivo_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ronda, j1, j2, resultado])


def jugar(rondas):
    puntos_j1 = puntos_j2 = empates = 0

    for ronda in range(1, rondas + 1):
        print(f"\n--- Ronda {ronda} de {rondas} ---")
        cuenta_regresiva()

        print("Esperando jugadas...")
        j1 = leer_jugada(opciones_j1)
        j2 = leer_jugada(opciones_j2)

        print(f"Jugador 1 eligió {j1} | Jugador 2 eligió {j2}")

        resultado = determinar_ganador(j1, j2)

        if resultado == "Empate":
            empates += 1
        elif resultado == "Jugador 1 gana":
            puntos_j1 += 1
        else:
            puntos_j2 += 1

        print(resultado)
        print(f"Marcador -> J1: {puntos_j1} | J2: {puntos_j2} | Empates: {empates}")

        guardar_csv(ronda, j1, j2, resultado)

    # Resultado final
    print("\n=== Resultado Final ===")
    print(f"Jugador 1: {puntos_j1} victorias")
    print(f"Jugador 2: {puntos_j2} victorias")
    print(f"Empates: {empates}")
    print(f"\nJugadas guardadas en '{archivo_csv}'.")
    print("Gracias por jugar.")


# Programa principal
if __name__ == "__main__":
    inicializar_csv()

    print("¿Cuántas rondas se quieren jugar?")
    rondas = int(input())

    jugar(rondas)
