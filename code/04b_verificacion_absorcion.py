"""
Script 04b: verificacion de la absorcion de efectos fijos.

Estima el Modelo 3 de dos maneras y compara los resultados:
  (A) incluyendo las 803 dummies de marca-tienda como columnas,
  (B) absorbiendolas por Frisch-Waugh-Lovell (centrado dentro del grupo).

Si el teorema es correcto, ambas deben dar los mismos coeficientes y,
tras la correccion de grados de libertad, los mismos errores estandar.

Correr:  python code/04b_verificacion_absorcion.py
"""

import time
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
df = pd.read_csv(RAIZ / "data" / "base_limpia.csv")
df["marca_tienda"] = (df["producto"].astype(str) + "-"
                      + df["tienda"].astype(str))
G = df["marca_tienda"].nunique()
n, k = len(df), 2
cl = {"cov_type": "cluster", "cov_kwds": {"groups": df["mercado"]}}

# ---------- (A) con las 803 dummies explicitas ------------------------
t0 = time.time()
D = pd.get_dummies(df["marca_tienda"], drop_first=True, dtype=float)
XA = sm.add_constant(pd.concat([df[["precio", "descuento"]], D], axis=1))
A = sm.OLS(df["y_logit"], XA).fit(**cl)
tA = time.time() - t0

# ---------- (B) absorbiendo por centrado ------------------------------
t0 = time.time()
COLS = ["y_logit", "precio", "descuento"]
dm = df[COLS] - df.groupby("marca_tienda")[COLS].transform("mean")
B = sm.OLS(dm["y_logit"], dm[["precio", "descuento"]]).fit(**cl)
factor = np.sqrt((n - k) / (n - k - G + 1))
tB = time.time() - t0

# ---------- comparacion -----------------------------------------------
print("efectos fijos absorbidos: %d   |   correccion gl: %.6f\n" % (G, factor))
print("%-12s %14s %14s %12s" % ("", "(A) dummies", "(B) absorcion", "diferencia"))
for v in ["precio", "descuento"]:
    print("%-12s %14.9f %14.9f %12.2e"
          % (v, A.params[v], B.params[v], A.params[v] - B.params[v]))
    print("%-12s %14.9f %14.9f %12.2e"
          % ("  (ee)", A.bse[v], B.bse[v] * factor,
             A.bse[v] - B.bse[v] * factor))

print("\nee de (B) SIN corregir gl: precio = %.9f  (subestimado)"
      % B.bse["precio"])
print("suma de residuos al cuadrado: A = %.6f | B = %.6f"
      % ((A.resid ** 2).sum(), (B.resid ** 2).sum()))
print("\ntiempo: (A) %.1f s   (B) %.2f s   ->  %.0fx mas rapido"
      % (tA, tB, tA / tB))
