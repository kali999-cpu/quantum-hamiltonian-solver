# Install libraries first if needed:
# pip install pymatgen pandas

from pymatgen.core import Element
import pandas as pd
import os

# Metals list
metals = [
    "Fe","Mo","Ru","Co","Ni","W","V",
    "Mn","Cr","Cu","Rh","Re","Os","Ir"
]

rows = []

for symbol in metals:
    el = Element(symbol)

    # count d electrons
    d_electrons = sum(e for n, orb, e in el.full_electronic_structure if orb == "d")

    rows.append({
        "element": symbol,
        "electronegativity": el.X,
        "atomic_radius": float(el.atomic_radius),
        "ionization_energy": el.ionization_energies[0],
        "electron_affinity": el.electron_affinity,
        "d_electrons": d_electrons,
        "group": el.group,
        "period": el.row,
        "melting_point": el.melting_point
    })

# Create dataframe
df = pd.DataFrame(rows)

# Save CSV file
file_path = "atomic_properties.csv"
df.to_csv(file_path, index=False)

print("CSV file saved at:", os.path.abspath(file_path))

# Read CSV again (to test)
df2 = pd.read_csv(file_path)

print("\nPreview of saved data:")
print(df2.head())