from pymatgen.core import Element
import pandas as pd

metals = ["Fe","Mo","Ru","Co","Ni","W","V",
          "Mn","Cr","Cu","Rh","Re","Os","Ir"]

rows = []
for symbol in metals:
    el = Element(symbol)

    # count d electrons
    d_electrons = sum(e for n,orb,e in el.full_electronic_structure if orb == "d")

    rows.append({
        "element": symbol,
        "electronegativity": el.X,
        "atomic_radius": float(el.atomic_radius),
        "ionization_energy": el.ionization_energies[0],
        "electron_affinity": el.electron_affinity,
        "d_electrons": d_electrons,
        "group": el.group,
        "period": el.row,
        "melting_point": el.melting_point,
    })

df = pd.DataFrame(rows)

# Save files
df.to_csv("atomic_properties.csv", index=False)
#df.to_excel("atomic_properties.xlsx", index=False)

print(df)