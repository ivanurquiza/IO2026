"""
Script 03: Ejercicio 3, modelo 2.
Logit agregado por MCO con precio, promocion y dummies por marca.

Especificacion:  ln(s_jt) - ln(s_0t) = beta_1*precio + beta_2*descuento
                                       + d_j + xi_jt
donde d_j son 11 efectos fijos, uno por marca (las 11 combinaciones
marca/tamano de la base).

Correr:  python code/03_logit_fe_marca.py
"""

import pandas as pd
import statsmodels.api as sm
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
df = pd.read_csv(RAIZ / "data" / "base_limpia.csv")

y = df["y_logit"]

# dummies por marca: 11 categorias -> 10 dummies + constante
# (drop_first=True evita la trampa de las variables dicotomicas)
D = pd.get_dummies(df["producto"], prefix="marca", drop_first=True, dtype=float)
X = sm.add_constant(pd.concat([df[["precio", "descuento"]], D], axis=1))

m2 = sm.OLS(y, X).fit(cov_type="cluster",
                      cov_kwds={"groups": df["mercado"]})
print(m2.summary())
