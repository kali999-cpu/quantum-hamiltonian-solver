# quantum-hamiltonian-solver
A research project focused on developing a novel computational approach to efficiently solve Hamiltonian equations in physics


🌍 PART 1 — THE MAIN USE & GOAL OF THIS PROJECT
The Single Most Important Problem This Solves
Right now, the world makes ammonia (NH₃) using the Haber-Bosch process — invented in 1909. It hasn't fundamentally changed since.
THE HABER-BOSCH REALITY TODAY
══════════════════════════════════════════════════════════════
  N₂  +  3H₂  →  2NH₃
  
  Conditions needed:     400–500°C temperature
                         150–300 atmospheres pressure
                         Iron catalyst (same since 1909)
                         
  Cost to the world:     2% of ALL global energy consumed
                         1.4% of ALL CO₂ emissions
                         $150 billion industry per year
                         
  Why we can't stop:     NH₃ → fertilizers → food
                         Without it, 50% of humans would
                         not have enough food to survive
══════════════════════════════════════════════════════════════
The dream: Do the same reaction at room temperature, using only electricity and air.
THE eNRR DREAM
══════════════════════════════════════════════════════════
  N₂  +  6H⁺  +  6e⁻  →  2NH₃
  
  Conditions needed:     Room temperature (25°C)
                         Normal pressure (1 atm)
                         Electricity from solar/wind
                         The right CATALYST
                         
  Problem:               We don't know what the right
                         catalyst is. There are millions
                         of possible materials.
                         
  Solution:              Use AI + ML to find it
══════════════════════════════════════════════════════════
This project's main goal = find that catalyst using computers instead of years of lab experiments.

🎯 PART 2 — THE THREE GOALS OF YOUR PROJECT
Goal 1 — Solve the Hamiltonian Faster (New Computational Methods)
What is the Hamiltonian?
Think of the Hamiltonian as the master equation of chemistry. Every single chemical reaction, every bond, every material property — all of it comes from solving one equation:
Ĥ ψ = E ψ

WHERE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Ĥ  =  Hamiltonian operator
         = Total energy of the system
         = Kinetic energy of electrons
           + attraction between electrons & nuclei
           + repulsion between electrons
           + repulsion between nuclei

  ψ  =  Wavefunction
         = Mathematical description of WHERE every
           electron is and what it is doing
         = Contains ALL chemistry information

  E  =  Energy eigenvalue
         = The number we actually want to know
         = Tells us: does N₂ stick to this surface?
           How strongly? Will it react?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Why is this hard?
Imagine you have a catalyst surface with 200 atoms. That means roughly 1,600 electrons. The wavefunction ψ depends on the position of ALL of them simultaneously. That's a mathematical object with 4,800 dimensions. No computer on Earth can store or solve this.
So scientists use approximations. The best practical approximation is called DFT (Density Functional Theory). But even DFT takes hours per calculation. If you want to test 1 million catalyst materials, that's 114 years of computing time.
The new method: Train a machine learning model to predict the OUTPUT of the Hamiltonian (the energies) without actually solving it. The ML model learns the patterns from a few thousand DFT calculations, then predicts the rest in milliseconds.
OLD WAY:           1 material → DFT (6 hours) → energy
NEW WAY:           1 material → ML model (0.001 seconds) → energy
                   
RESULT:            Screen 1,000,000 materials in one day
                   instead of 114 years

Goal 2 — DFT Studies (Calculate Ground Truth)
DFT = Density Functional Theory
This is the bridge between quantum physics and real chemistry. Instead of solving for ψ (4,800 dimensions), DFT proves you only need the electron density ρ(r) — how many electrons are at each point in space — which has only 3 dimensions.
What DFT actually tells you for eNRR:
For a catalyst to work, nitrogen must go through a series of steps on the surface. Each step either releases or costs energy. DFT calculates the energy of every step:
NITROGEN REDUCTION PATHWAY ON A CATALYST SURFACE
═══════════════════════════════════════════════════════════════════
                    
  N₂(gas)                 Step 0: N₂ arrives
      ↓  ΔG₁              
  *N₂ (adsorbed)          Step 1: N₂ sticks to surface
      ↓  ΔG₂              
  *NNH                    Step 2: First H added (hardest step!)
      ↓  ΔG₃              
  *NNH₂                   Step 3: Second H added
      ↓  ΔG₄              
  *N  +  NH₃(gas)         Step 4: First NH₃ released
      ↓  ΔG₅              
  *NH                     Step 5: N gets H
      ↓  ΔG₆              
  *NH₂                    Step 6: NH₂ forms
      ↓  ΔG₇              
  NH₃(gas)                Step 7: Second NH₃ released

═══════════════════════════════════════════════════════════════════
  DFT calculates each ΔG
  The LARGEST positive ΔG = potential-determining step
  This tells you how much voltage you need to apply
  Lower voltage needed = better catalyst
═══════════════════════════════════════════════════════════════════
What makes a good catalyst?
The Sabatier Principle — the binding must be "just right":
TOO WEAK BINDING          JUST RIGHT              TOO STRONG BINDING
════════════════          ══════════              ══════════════════
N₂ doesn't stick          N₂ sticks well          N₂ sticks but never
to surface                reacts & releases        lets go (poisoning)
No reaction               NH₃ easily              No NH₃ produced

← ── ── ── ── ── VOLCANO PEAK ── ── ── ── ── →
              ΔG_N* ≈ −0.35 eV
              (optimal N binding)

Goal 3 — AI/ML Model (Scale Up Discovery)
The ML model does three things:
Thing 1 — Learn from DFT data
Train on the DFT calculations you already have. The model learns: "when atoms have these electronic properties, the N binding energy is approximately this value."
Thing 2 — Predict for new materials instantly
For any new material you haven't calculated, the model predicts its N binding energy in milliseconds.
Thing 3 — Rank and identify the best catalysts
Use the volcano plot to identify which predicted materials sit closest to the optimal point. These are your candidates for experimental testing.

⚗️ PART 3 — THE IMPORTANCE (Why This Matters to the World)
Scale of Impact
IF THIS PROJECT SUCCEEDS AND eNRR BECOMES PRACTICAL:
═══════════════════════════════════════════════════════════════════════
  
  FOOD:        Replace carbon-intensive fertilizer production
               Feed 8 billion people with zero-carbon agriculture
               Bring fertilizer production to villages in Africa/Asia
               (currently impossible without massive industrial plants)
  
  ENERGY:      Ammonia = liquid hydrogen carrier
               Store excess solar/wind energy as NH₃
               Ship green energy globally as liquid fuel
               Convert back to electricity/H₂ at destination
               Solve the energy storage problem
  
  CLIMATE:     Eliminate 450 million tonnes of CO₂ per year
               One of the SINGLE BIGGEST climate interventions possible
               More impactful than all electric cars combined
  
  ECONOMY:     $150 billion ammonia industry replaced by green tech
               Massive new industry in renewable energy countries
               Jobs in computational chemistry, materials science, AI
  
  SCIENCE:     New methods for solving quantum equations
               Transfer to drug discovery, battery materials,
               solar cells, superconductors — same technique
               applies everywhere in materials science
═══════════════════════════════════════════════════════════════════════
Importance to Science Specifically
HAMILTONIAN METHODS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Importance 1:  Faster drug discovery
               Same ML-Hamiltonian methods apply to
               predicting how drugs bind to proteins
               
Importance 2:  Battery materials
               Find better lithium/sodium battery
               electrode materials for EVs
               
Importance 3:  Superconductors
               Predict room-temperature superconductors
               (would revolutionize electronics & energy)
               
Importance 4:  Fundamental physics
               Deepen understanding of quantum many-body
               problem — one of physics' grand challenges

DFT STUDIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Importance 1:  Explain WHY current catalysts work
               Understand the electronic mechanism
               
Importance 2:  Guide experimentalists
               Tell lab scientists exactly which
               materials to synthesize and test
               
Importance 3:  Predict stability
               Will the catalyst corrode in water?
               Will it survive the reaction conditions?
               
Importance 4:  Replace expensive experiments
               One DFT calculation costs $0.01 of compute
               One lab synthesis/test costs $500–$5000

ML FOR CATALYSTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Importance 1:  Scale — test millions not hundreds
               
Importance 2:  Speed — days not decades
               
Importance 3:  Interpretability via SHAP
               Tells you WHICH atomic property matters most
               → guides design of even better catalysts
               
Importance 4:  Active learning
               Model learns which experiments to run next
               → minimizes wasted effort
