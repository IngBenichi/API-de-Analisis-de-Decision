from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class Matriz:
    def __init__(self, decisores: int, criterios: int) -> None:
        self.decisores = decisores 
        self.criterios = criterios 
        self.categorias = {}
        self.matriz = None
        self.criterios_beneficio = []
        self.pesos = []

    def definirCalificaciones(self) -> None:
        self.categorias = {
            'VL': (1, 1, 3),
            'L': (1, 3, 5),
            'M': (3, 5, 7),
            'H': (5, 7, 9),
            'VH': (7, 9, 9)
        }

    def definirCriteriosBeneficio(self, criterios_beneficio: list):
        self.criterios_beneficio = criterios_beneficio
        
    def definirPesos(self, pesos: list):
        self.pesos = pesos

    def cargarMatriz(self, entrada: str) -> None:
        self.definirCalificaciones()
        self.matriz = np.empty((self.decisores, self.criterios), dtype=object)
        parametros = entrada.split()

        if len(parametros) != self.decisores * self.criterios:
            raise ValueError('El número de elementos debe ser igual a decisores * criterios ({} en total)'.format(self.decisores * self.criterios))

        k = 0
        for i in range(self.decisores):
            for j in range(self.criterios):
                if k < len(parametros):
                    categoria = parametros[k]
                    self.matriz[i, j] = self.categorias.get(categoria, (0, 0, 0))
                    k += 1
                else:
                    self.matriz[i, j] = self.categorias.get('VL')
        
        return self.matriz

    def solucion_ideal(self):
        self.sip = []
        self.sin = []
        
        for j in range(self.criterios):
            columna = [self.matriz[i, j] for i in range(self.decisores)]
            if self.criterios_beneficio[j]:
                self.sip.append(max(col[2] for col in columna))
                self.sin.append(min(col[0] for col in columna))
            else:
                self.sip.append(min(col[0] for col in columna))
                self.sin.append(max(col[2] for col in columna))
        return self.sip, self.sin

    def calcular_proximidad(self):
        self.proximidad = []
        
        for j in range(self.criterios):
            distancia_sip = sum((self.matriz[i, j][2] - self.sip[j]) ** 2 for i in range(self.decisores)) ** 0.5
            distancia_sin = sum((self.matriz[i, j][0] - self.sin[j]) ** 2 for i in range(self.decisores)) ** 0.5
            proximidad_i = distancia_sin / (distancia_sip + distancia_sin) if (distancia_sip + distancia_sin) > 0 else 0
            self.proximidad.append(proximidad_i)
        
        return self.proximidad

    def ordenar_alternativas(self):
        indices = list(range(self.criterios))
        indices.sort(key=lambda i: self.proximidad[i], reverse=True)
        
        return indices, self.proximidad

class MatrizRequest(BaseModel):
    decisores: int
    criterios: int
    entrada: str
    criterios_beneficio: List[bool]
    pesos: List[float]

    @validator("criterios_beneficio")
    def validar_criterios_beneficio(cls, v, values):
        criterios = values.get("criterios")
        if criterios is not None and len(v) != criterios:
            raise ValueError("La cantidad de criterios de beneficio debe coincidir con el número de criterios.")
        return v

    @validator("pesos")
    def validar_pesos(cls, v, values):
        criterios = values.get("criterios")
        if criterios is not None and len(v) != criterios:
            raise ValueError("La cantidad de pesos debe coincidir con el número de criterios.")
        return v

@app.post("/matriz/")
async def crear_y_procesar_matriz(request: MatrizRequest):
    matriz = Matriz(request.decisores, request.criterios)
    try:
        matriz_original = matriz.cargarMatriz(request.entrada)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    matriz.definirCriteriosBeneficio(request.criterios_beneficio)
    matriz.definirPesos(request.pesos)

    # Realizar las operaciones
    sip, sin = matriz.solucion_ideal()
    proximidad = matriz.calcular_proximidad()
    indices, proximidad_ordenada = matriz.ordenar_alternativas()
    
    return {
        "matriz_original": matriz_original.tolist(),
        "solucion_ideal_positiva": sip,
        "solucion_ideal_negativa": sin,
        "proximidad": proximidad,
        "orden_alternativas": indices,
        "proximidad_ordenada": proximidad_ordenada
    }
