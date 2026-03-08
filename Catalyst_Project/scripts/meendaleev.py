from mendeleev import element
import pandas as pd

metals = ["Fe","Mo","Ru","Co","Ni","W","V","Mn","Cr","Cu","Rh","Re","Os","Ir"]

rows = []

for m in metals:
    el = element(m)

    rows.append({
        "Element": el.symbol,
        "Electronegativity": el.electronegativity(),
        "Atomic_radius": el.atomic_radius,
        "Electron_affinity": el.electron_affinity,
        "d_electrons": el.ec.conf.get("d",0),
        "Covalent_radius": el.covalent_radius
    })

df = pd.DataFrame(rows)

df.to_csv("catalyst_atomic_properties.csv", index=False)

print(df)