"""
Script 10: Ejercicio 4b. Elasticidades-precio en la tienda 9, semana 10.

Compara la matriz de elasticidades 11x11 del modelo BLP contra la del
logit (que se obtiene fijando todos los coeficientes de dispersion en
cero). El punto del inciso: en el logit cada columna de la matriz es
constante -es la IIA-, mientras que en BLP varia, porque los productos
de una misma marca madre sustituyen mas entre si.

Correr:  python code/10_blp_elasticidades.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyblp

pyblp.options.verbose = False
RAIZ = Path(__file__).resolve().parent.parent
MERCADO = "9-10"

# ---------- cargar la estimacion 4a ------------------------------------
with open(RAIZ / "output" / "blp_medios.pkl", "rb") as f:
    res_blp = pickle.load(f)

# ---------- elasticidades BLP ------------------------------------------
# compute_elasticities devuelve, para cada mercado, la matriz cuyo
# elemento (a,c) es la elasticidad del share de a ante el precio de c.
elast = res_blp.compute_elasticities()
prod = pd.read_csv(RAIZ / "data" / "blp_productos.csv")
mask = (prod["market_ids"] == MERCADO).to_numpy()
productos = prod.loc[mask, "product_ids"].to_numpy()
E_blp = pd.DataFrame(elast[mask], index=productos, columns=productos)

# ---------- elasticidades logit ----------------------------------------
# El logit es el caso particular con toda la dispersion en cero. Se
# reestima el mismo problema con sigma=0 y pi=0 y se computan sus
# elasticidades, para que ambas matrices sean comparables.
problema = res_blp.problem
res_logit = problema.solve(
    sigma=np.zeros((5, 5)),
    pi=np.zeros((5, 1)),
    method="1s",
    optimization=pyblp.Optimization("return"),   # no optimiza: evalua en 0
)
E_logit = pd.DataFrame(
    res_logit.compute_elasticities()[mask], index=productos, columns=productos)

# ---------- reporte -----------------------------------------------------
np.set_printoptions(precision=3, suppress=True)
print("=== Elasticidades BLP, mercado %s ===" % MERCADO)
print(E_blp.round(3).to_string())
print("\n=== Elasticidades logit, mercado %s ===" % MERCADO)
print(E_logit.round(3).to_string())

# La IIA se ve en la variacion DENTRO de cada columna (fuera de la
# diagonal): en el logit es ~0, en BLP es positiva.
n = len(productos)
fuera_diag = ~np.eye(n, dtype=bool)
comp = pd.DataFrame({
    "logit_sd_col": [E_logit.values[fuera_diag[:, k], k].std() for k in range(n)],
    "blp_sd_col":   [E_blp.values[fuera_diag[:, k], k].std() for k in range(n)],
}, index=productos)
print("\n=== Desvio de las cruzadas dentro de cada columna ===")
print("(logit ~0 = IIA; BLP > 0 = sustitucion diferenciada)")
print(comp.round(4).to_string())

print("\nelasticidad propia media: BLP %.3f | logit %.3f"
      % (np.diag(E_blp).mean(), np.diag(E_logit).mean()))