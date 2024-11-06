---

# API de Análisis de Decisión

Esta API permite crear y procesar matrices de decisión en función de ciertos criterios y sus respectivos pesos. Permite definir una matriz de calificaciones categorizadas, calcular soluciones ideales y ordenar alternativas en base a la proximidad.

### Requisitos
Para usar esta API, asegúrate de tener instalados los siguientes paquetes:
```bash
pip install fastapi numpy pydantic uvicorn
```

### Ejecución
Para ejecutar la aplicación en el entorno local:
```bash
uvicorn main:app --reload
```

### Middleware
La API usa `CORSMiddleware` para permitir solicitudes de cualquier origen. Esto es útil en entornos de desarrollo, ya que permite el acceso desde diferentes dominios:
- `allow_origins=["*"]`: permite todos los orígenes.
- `allow_methods=["*"]`: permite todos los métodos HTTP.
- `allow_headers=["*"]`: permite todos los encabezados.

---

## Modelo de datos

### `MatrizRequest`
Esquema de solicitud para la creación y procesamiento de una matriz:
- **decisores**: `int` - Número de personas que toman decisiones.
- **criterios**: `int` - Número de criterios a evaluar.
- **entrada**: `str` - Cadena que contiene las calificaciones categorizadas para cada criterio y decisor.
- **criterios_beneficio**: `List[bool]` - Lista de valores booleanos donde `true` indica que un criterio es de beneficio y `false` indica que es de costo.
- **pesos**: `List[float]` - Lista de pesos para cada criterio.

**Validaciones**:
- `criterios_beneficio` y `pesos` deben tener la misma longitud que el número de criterios definidos.

---

## Endpoints

### `POST /matriz/`
Este endpoint permite crear una matriz, definir sus calificaciones, criterios de beneficio y pesos, y realizar cálculos sobre las alternativas.

#### Solicitud
- **Ruta**: `/matriz/`
- **Método**: `POST`
- **Cuerpo de la solicitud**: Un objeto `MatrizRequest` en formato JSON.

#### Respuesta
- **matriz_original**: Matriz creada y calificada según las categorías dadas.
- **solucion_ideal_positiva** (`sip`): Lista de valores que representan la mejor solución para cada criterio.
- **solucion_ideal_negativa** (`sin`): Lista de valores que representan la peor solución para cada criterio.
- **proximidad**: Lista de valores que indican la proximidad de cada criterio a las soluciones ideales.
- **orden_alternativas**: Orden de los criterios en función de su proximidad a la solución ideal.
- **proximidad_ordenada**: Lista de proximidades ordenadas de mayor a menor.

#### Ejemplo de solicitud
```json
{
    "decisores": 3,
    "criterios": 2,
    "entrada": "VL L M H VL M",
    "criterios_beneficio": [true, false],
    "pesos": [0.3, 0.2]
}

```

#### Ejemplo de respuesta
```json
{
{
  "matriz_original": [
    [
      [
        1,
        1,
        3
      ],
      [
        1,
        3,
        5
      ]
    ],
    [
      [
        3,
        5,
        7
      ],
      [
        5,
        7,
        9
      ]
    ],
    [
      [
        1,
        1,
        3
      ],
      [
        3,
        5,
        7
      ]
    ]
  ],
  "solucion_ideal_positiva": [
    7,
    1
  ],
  "solucion_ideal_negativa": [
    1,
    9
  ],
  "proximidad": [
    0.2612038749637414,
    0.5
  ],
  "orden_alternativas": [
    1,
    0
  ],
  "proximidad_ordenada": [
    0.2612038749637414,
    0.5
  ]
}
```

---

### Explicación de Clases y Métodos

1. **Clase `Matriz`**: Define una matriz de decisión con:
   - Calificaciones para cada criterio y decisor en categorías como "VL" (Muy Bajo) hasta "VH" (Muy Alto).
   - Criterios de beneficio para evaluar qué criterios maximizan o minimizan la puntuación.
   - Cálculo de la solución ideal positiva y negativa para cada criterio.
   - Cálculo de la proximidad de cada criterio a la solución ideal y su orden.

2. **Métodos Importantes**:
   - **definirCalificaciones**: Configura las categorías de calificaciones.
   - **definirCriteriosBeneficio**: Define cuáles criterios son de beneficio.
   - **cargarMatriz**: Carga la matriz de calificaciones en base a la entrada dada.
   - **solucion_ideal**: Calcula la solución ideal positiva y negativa para cada criterio.
   - **calcular_proximidad**: Calcula la proximidad de cada criterio a la solución ideal.
   - **ordenar_alternativas**: Ordena las alternativas en función de su proximidad a la solución ideal.
