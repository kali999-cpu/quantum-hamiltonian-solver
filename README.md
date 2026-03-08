# quantum-hamiltonian-solver
A research project focused on developing a novel computational approach to efficiently solve Hamiltonian equations in physics


Here is a **clean, professional GitHub README** written in a **standard research-project format**. It keeps your scientific depth but organizes it the way **GitHub, open-source, and research repositories normally expect**.

You can copy this directly into a file called **`README.md`** in your repository.

---

# AI-Accelerated Discovery of Catalysts for Electrochemical Nitrogen Reduction (eNRR)

*A computational research project combining quantum physics, density functional theory (DFT), and machine learning to accelerate catalyst discovery for sustainable ammonia production.*

---

# Overview

This project develops a **novel computational workflow for solving Hamiltonian-derived energy predictions using machine learning** in order to accelerate the discovery of catalysts for the **electrochemical nitrogen reduction reaction (eNRR)**.

The work integrates:

* **Quantum physics (Hamiltonian formulation)**
* **Density Functional Theory (DFT) simulations**
* **Machine learning models**
* **Materials discovery workflows**

The goal is to **replace slow quantum calculations with fast AI predictions**, enabling the screening of **millions of candidate catalysts**.

---

# The Global Problem

Ammonia (NH₃) production currently relies on the **Haber-Bosch process**, developed in 1909.

### Haber-Bosch Reaction

```
N₂ + 3H₂ → 2NH₃
```

Operating conditions:

* Temperature: **400–500°C**
* Pressure: **150–300 atm**
* Catalyst: **iron-based**

### Global Impact

| Metric                    | Value                                                         |
| ------------------------- | ------------------------------------------------------------- |
| Global energy consumption | ~2%                                                           |
| Global CO₂ emissions      | ~1.4%                                                         |
| Industry value            | ~$150 billion                                                 |
| Importance                | Supports fertilizer production for ~50% of global food supply |

Despite its importance, Haber-Bosch is **highly energy intensive and carbon intensive**.

---

# The Alternative: Electrochemical Nitrogen Reduction (eNRR)

A sustainable alternative is the **electrochemical nitrogen reduction reaction**:

```
N₂ + 6H⁺ + 6e⁻ → 2NH₃
```

Operating conditions:

* **Room temperature (25°C)**
* **Atmospheric pressure**
* **Renewable electricity**

The key challenge:

> **Finding an efficient catalyst.**

Millions of possible materials exist, but experimentally testing them is impractical.

This project uses **AI-driven materials discovery** to identify promising candidates.

---

# Scientific Background

## Hamiltonian Equation

All chemical properties originate from solving the **quantum mechanical Hamiltonian equation**:

```
Ĥ ψ = E ψ
```

Where:

| Symbol | Meaning                                    |
| ------ | ------------------------------------------ |
| Ĥ      | Hamiltonian operator (total system energy) |
| ψ      | wavefunction describing electron behavior  |
| E      | energy eigenvalue                          |

The Hamiltonian includes:

* electron kinetic energy
* electron–nucleus attraction
* electron–electron repulsion
* nucleus–nucleus repulsion

For realistic materials, solving this equation exactly is **computationally impossible**.

---

## Density Functional Theory (DFT)

Density Functional Theory reduces the complexity by replacing the wavefunction ψ with the **electron density ρ(r)**.

Instead of thousands of dimensions, the problem becomes **three-dimensional**.

DFT enables calculation of:

* adsorption energies
* reaction barriers
* catalyst surface properties
* electronic structure descriptors

However, **DFT is still computationally expensive**, often requiring **hours per calculation**.

Screening millions of materials using DFT alone would take **decades of compute time**.

---

# Research Approach

This project introduces a **three-stage computational pipeline**.

---

## 1. DFT Data Generation

DFT calculations provide **ground-truth energy values** for a set of catalyst materials.

For eNRR, the nitrogen reduction pathway includes several reaction intermediates:

```
N₂(g)
↓
*N₂
↓
*NNH
↓
*NNH₂
↓
*N + NH₃
↓
*NH
↓
*NH₂
↓
NH₃
```

Each step has a **Gibbs free energy change (ΔG)**.

The **largest positive ΔG** determines the **potential-determining step**, which defines catalyst performance.

---

## 2. Machine Learning Model

A machine learning model is trained using DFT-derived data.

Input features include:

* electronegativity
* atomic radius
* ionization energy
* electron affinity
* d-band center
* work function
* surface energy
* magnetic moment
* nitride formation energy

The model learns relationships between **electronic structure descriptors** and **nitrogen adsorption energy (ΔG_N)**.

After training, the model can predict catalyst energies in **milliseconds**.

---

## 3. Catalyst Screening

Predicted adsorption energies are analyzed using the **volcano plot principle**.

According to the **Sabatier principle**, the optimal catalyst binds nitrogen **neither too weakly nor too strongly**.

```
Weak Binding        Optimal Binding        Strong Binding
No adsorption   ←   Volcano Peak   →   Catalyst poisoning
```

The optimal nitrogen adsorption energy is approximately:

```
ΔG_N ≈ −0.35 eV
```

Materials predicted near this value are **top catalyst candidates**.

---

# Computational Workflow

```
DFT Calculations
        │
        ▼
Dataset Construction
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Machine Learning Training
        │
        ▼
Feature Importance Analysis
        │
        ▼
Catalyst Screening
```

---

# Project Structure

```
project/
│
├── data/
│   ├── raw_dft_data.csv
│   └── master_dataset.csv
│
├── scripts/
│   ├── Step1_DFT_Data.py
│   ├── Step2_Create_Dataset.py
│   ├── Step3_EDA.py
│   └── Step4_ML_Model.py
│
├── plots/
│   ├── volcano_plot.png
│   ├── feature_importance.png
│   └── predictions_vs_dft.png
│
├── models/
│   ├── best_model.pkl
│   └── scaler.pkl
│
└── README.md
```

---

# Technologies Used

* **Python**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **Matplotlib**
* **Density Functional Theory data**
* **Machine learning regression models**

---

# Scientific Impact

## Sustainable Agriculture

Green ammonia could replace carbon-intensive fertilizer production.

This would support **global food systems with lower environmental impact**.

---

## Renewable Energy Storage

Ammonia is a **liquid hydrogen carrier**, enabling:

* long-distance energy transport
* seasonal energy storage
* renewable fuel systems

---

## Climate Impact

Potential reduction of:

**~450 million tonnes of CO₂ per year**

This would make ammonia synthesis one of the **largest climate mitigation opportunities**.

---

## Materials Discovery

The ML-accelerated Hamiltonian approach can also be applied to:

* battery materials
* catalysts for CO₂ reduction
* drug discovery
* superconductors
* photovoltaic materials

---

# Future Work

Potential improvements include:

* Active learning workflows
* Graph neural networks for atomic structures
* Integration with high-throughput DFT databases
* Experimental validation of predicted catalysts

---

# Author

Research project in **Computational Materials Science and Data Analytics**, focusing on **AI-driven catalyst discovery** for sustainable chemical processes.

---

# License

MIT License





