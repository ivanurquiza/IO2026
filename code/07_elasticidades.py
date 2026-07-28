"""
Script 07: Ejercicio 3, inciso 6.
Elasticidades-precio propias del logit para los modelos 1, 2 y 3.

Formula analitica (ver Notas de Clase 03, seccion 3.3):
    propia   eta_jj = alpha * p_jt * (1 - s_jt)
    cruzada  eta_ac = -alpha * p_ct * s_ct

Se evalua en cada una de las 38.544 observaciones con el precio y el
share efectivamente observados, y luego se promedia por marca sobre los
3.504 mercados.

Correr:  python code/07_elasticidades.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
df = pd.read_csv(RAIZ / "data" / "base_limpia.csv")
df["const"] = 1.0
df["marca_tienda"] = (df["producto"].astype(str) + "-"
                      + df["tienda"].astype(str))
CL = dict(cov_type="cluster", cov_kwds={"groups": df["mercado"]})

# ---------- alphas de los tres modelos por MCO -------------------------
X1 = sm.add_constant(df[["precio", "descuento"]])
a1 = sm.OLS(df["y_logit"], X1).fit(**CL).params["precio"]

D = pd.get_dummies(df["producto"], prefix="m", drop_first=True, dtype=float)
X2 = sm.add_constant(pd.concat([df[["precio", "descuento"]], D], axis=1))
a2 = sm.OLS(df["y_logit"], X2).fit(**CL).params["precio"]

C = ["y_logit", "precio", "descuento"]
dm = df[C] - df.groupby("marca_tienda")[C].transform("mean")
a3 = sm.OLS(dm["y_logit"], dm[["precio", "descuento"]]).fit(**CL).params["precio"]

# ---------- alpha del mejor IV (Hausman, modelo 3), como referencia ----
g = df.groupby(["producto", "semana"])["precio"]
df["z"] = (g.transform("sum") - df["precio"]) / (g.transform("size") - 1)
C2 = C + ["z"]
dm2 = df[C2] - df.groupby("marca_tienda")[C2].transform("mean")
aIV = IV2SLS(dm2["y_logit"], dm2[["descuento"]], dm2["precio"],
             dm2["z"]).fit(cov_type="clustered",
                           clusters=df["mercado"]).params["precio"]

ALPHAS = {"Modelo 1": a1, "Modelo 2": a2, "Modelo 3": a3,
          "IV Hausman (M3)": aIV}

# ---------- elasticidades ---------------------------------------------
tabla = pd.DataFrame(index=range(1, 12))
for nombre, a in ALPHAS.items():
    eta = a * df["precio"] * (1 - df["share"])
    tabla[nombre] = eta.groupby(df["producto"]).mean()

print("alphas:", {k: round(v, 4) for k, v in ALPHAS.items()})
print("\nElasticidad-precio propia promedio por marca:")
print(tabla.round(3).to_string())
print("\nPromedio general:")
print(tabla.mean().round(3).to_string())

# ---------- referencia: elasticidad implicada por los margenes ---------
lerner = ((df["precio"] - df["costo"]) / df["precio"])
print("\nMargen (p-c)/p observado: media %.3f" % lerner.mean())
print("Elasticidad implicada por Lerner (1/margen): %.2f" % (-1 / lerner.mean()))
print("\nMarcas con |elasticidad| < 1:")
print((tabla.abs() < 1).sum().to_string(), "de 11")

# ---------- elasticidades cruzadas promedio ----------------------------
print("\nElasticidad cruzada promedio:")
for nombre, a in ALPHAS.items():
    print("  %-16s %.5f" % (nombre, (-a * df["precio"] * df["share"]).mean()))
