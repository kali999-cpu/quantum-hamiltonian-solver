from mp_api.client import MPRester

API_KEY = "5e1O3HkIrFlCG9VRSqOMZL2EwBStKA0x"

with MPRester(API_KEY) as mpr:
    structure = mpr.get_structure_by_material_id("mp-149")

    print("Formula:", structure.formula)
    print("Lattice:", structure.lattice)