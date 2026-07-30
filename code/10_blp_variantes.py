"""
Script 11: Ejercicio 4a. Comparacion de especificaciones.

Estima el mismo modelo bajo variantes que difieren en un solo aspecto,
para poder justificar la eleccion final por el valor de la funcion GMM:

  A) X1 con las 4 dummies de marca madre (lectura literal de la consigna)
     frente a la base, que absorbe las 11 dummies de producto.

Cada variante se corre con los dos valores iniciales.

Correr:  python code/11_blp_variantes.py
"""

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyblp

RAIZ = Path(__file__).resolve().parent.parent
SALIDA = RAIZ / "output"
SALIDA.mkdir(exist_ok=True)

pyblp.options.verbose = False
prod = pd.read_csv(RAIZ / "data" / "blp_productos.csv")
agen = pd.read_csv(RAIZ / "data" / "blp_agentes.csv")

X2 = pyblp.Formulation("0 + marca1 + marca2 + marca3 + marca4 + prices")
AG = pyblp.Formulation("0 + ingreso")

# --- las dos formulaciones de X1 que se comparan -----------------------
ESPECIFICACIONES = {
    "11 dummies de producto (absorbidas)":
        pyblp.Formulation("0 + prices + promo", absorb="C(product_ids)"),
    "4 dummies de marca":
        pyblp.Formulation("0 + prices + promo + marca1 + marca2 + marca3 + marca4"),
}

CASI_CERO = 1e-2
INICIALES = {
    "cerca_de_cero": (np.diag([CASI_CERO] * 4 + [0.0]),
                      np.array([[0.0], [0.0], [0.0], [0.0], [-CASI_CERO]])),
    "unos":          (np.diag([1.0, 1.0, 1.0, 1.0, 0.0]),
                      np.array([[0.0], [0.0], [0.0], [0.0], [-0.05]])),
}

filas = []
for etiqueta, X1 in ESPECIFICACIONES.items():
    problema = pyblp.Problem((X1, X2), prod, AG, agen)
    for nombre, (sigma0, pi0) in INICIALES.items():
        print("\n--- %s | inicial: %s ---" % (etiqueta, nombre), flush=True)
        t = time.time()
        res = problema.solve(
            sigma=sigma0, pi=pi0, method="1s",
            optimization=pyblp.Optimization("l-bfgs-b", {"gtol": 1e-5}),
        )
        s = np.diag(res.sigma)
        filas.append({
            "X1": etiqueta, "inicial": nombre,
            "GMM": res.objective.item(),
            "sigma1": s[0], "sigma2": s[1], "sigma3": s[2], "sigma4": s[3],
            "pi_ingreso": res.pi[4, 0],
            "beta_precio": res.beta[0, 0],
            "min": (time.time() - t) / 60,
        })
        print(pd.DataFrame(filas[-1:]).round(4).to_string(index=False), flush=True)

tabla = pd.DataFrame(filas)
tabla.to_csv(SALIDA / "blp_variantes.csv", index=False)
print("\n" + "=" * 90)
print(tabla.round(4).to_string(index=False))
