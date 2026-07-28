"""
Script 04: Ejercicio 3, modelo 3.
Logit por MCO con precio, promocion y efectos fijos de marca-tienda.

Especificacion:  ln(s_jt) - ln(s_0t) = beta_1*precio + beta_2*descuento
                                       + d_{j,tienda} + xi_jt

Son 11 marcas x 73 tiendas = 803 efectos fijos. En lugar de incluirlos
como 803 columnas de dummies, los absorbemos por el teorema de
Frisch-Waugh-Lovell: restamos a cada variable su media dentro del grupo
marca-tienda y estimamos la regresion sobre los datos centrados. El
resultado es identico y el costo computacional es una fraccion.

Correr:  python code/04_logit_fe_marca_tienda.py
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
df = pd.read_csv(RAIZ / "data" / "base_limpia.csv")

# --- grupo de efecto fijo: la interaccion marca x tienda ---
df["marca_tienda"] = (df["producto"].astype(str) + "-"
                      + df["tienda"].astype(str))
G = df["marca_tienda"].nunique()
print("efectos fijos marca-tienda:", G)

# --- absorcion: restar la media del grupo a cada variable -------------
COLS = ["y_logit", "precio", "descuento"]
medias = df.groupby("marca_tienda")[COLS].transform("mean")
dm = df[COLS] - medias          # dm = "demeaned" (centrado dentro del grupo)

# sin constante: el centrado ya elimina cualquier intercepto
m3 = sm.OLS(dm["y_logit"], dm[["precio", "descuento"]]).fit(
    cov_type="cluster", cov_kwds={"groups": df["mercado"]})

# --- correccion de grados de libertad ---------------------------------
# statsmodels no sabe que absorbimos G parametros, asi que sus errores
# estandar quedan subestimados. El factor sqrt((n-k)/(n-k-G+1)) los
# devuelve al valor que se obtendria incluyendo las 803 dummies.
n, k = len(df), 2
factor = np.sqrt((n - k) / (n - k - G + 1))
ee = m3.bse * factor

print("\n--- Modelo 3 ---")
for v in ["precio", "descuento"]:
    print("%-10s coef = %8.4f   ee = %.4f" % (v, m3.params[v], ee[v]))

# R2 respecto del modelo completo (incluyendo lo explicado por los EF)
sst = ((df["y_logit"] - df["y_logit"].mean()) ** 2).sum()
ssr = (m3.resid ** 2).sum()
print("\nR2 (modelo completo, con EF) = %.4f" % (1 - ssr / sst))
print("R2 within (solo precio y promocion) = %.4f" % m3.rsquared)
print("Observaciones:", n)
