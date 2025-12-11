# Documentacion de Recogida de Datos

**Alumno:** ___Lili Wu, Javier Pasamar, Diana Palacios___

## Formato del CSV

Tu archivo `data/partidas.csv` debe tener **minimo** estas columnas:

| Columna | Descripcion | Ejemplo |
|---------|-------------|---------|
| `numero_ronda` | Numero de la ronda (1, 2, 3...) | 1 |
| `jugada_j1` | Jugada del jugador 1 | piedra |
| `jugada_j2` | Jugada del jugador 2 (oponente) | papel |

### Ejemplo de CSV minimo:

```csv
numero_ronda,jugada_j1,jugada_j2
1,piedra,papel
2,tijera,piedra
3,papel,papel
4,piedra,tijera
...
```

---

## Como recogiste los datos?

Marca con [x] el metodo usado y describe brevemente:

### Metodo de recogida:

- [x] **Programa propio**: Cree un programa para jugar y guardar datos
- [ ] **Manual**: Jugue partidas y apunte los resultados a mano
- [ ] **Aplicacion/Web externa**: Use una app y exporte los datos
- [ ] **Otro**: _________________

### Descripcion del proceso:

```

He creado un programa en la que te pregunta cuantas rondas quieres, y por cada ronda que pase, se guarda ronda por ronda 
la jugada en el archivo jugadas.csv



```
---

## Datos adicionales capturados

Si capturaste datos extra ademas de los basicos, marcalos aqui:

- [ ] `tiempo_reaccion_ms` - Tiempo que tardo el jugador en responder
- [ ] `timestamp` - Fecha/hora de cada jugada
- [ ] `sesion` - ID de sesion de juego
- [x] `resultado` - victoria/derrota/empate
- [ ] Otro: _________________

### Descripcion de datos adicionales:

```
(Si capturaste datos extra, explica aqui por que y como los usas)
Compruebo las veces que ha ganado y de que manera ha ganado el jugador.

```

---

## Estadisticas del dataset

- **Total de rondas:** _476_
- **Numero de sesiones/partidas:** _2_
- **Contra cuantas personas diferentes:** __1__

### Tipo de IA:

- [x] **IA Especifica**: Entrenada para ganar a UNA persona concreta
  - Nombre/identificador del oponente: ___Javier___
- [ ] **IA General**: Entrenada para ganar a cualquier oponente
