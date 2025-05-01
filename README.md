# 🚀 HAPS-Based MIMO Channel Modeling and Metaheuristic Optimization

This repository provides a comprehensive framework for simulating the performance of a **High Altitude Platform Station (HAPS)** using a **MIMO-based communication system** and solving the **HAPS placement optimization problem** via various **metaheuristic algorithms**.

---

## 📁 Folder Structure

```
.
├── env/                    # Main channel model components
│   ├── ChannelModel.py
│   ├── Map.py
│   ├── MIMOconfig.py
│   └── ...
├── Area Distribution/      # Area configuration CSVs
│   └── clustered_areas_r_*.csv
├── channel_model_demo.ipynb # Notebook demonstrating full pipeline
├── requirements.txt        # Python dependency list
└── README.md               # This file
```

---

## 🌐 1. Overview

In each time slot, the HAPS covers a selection of ground **areas**, each containing numerous users. Instead of modeling each user channel explicitly, the system approximates each area’s **effective channel** via **Monte Carlo sampling** of user locations.

This approach captures:
- User diversity (indoor/outdoor)
- Propagation conditions (LoS/NLoS)
- Environmental heterogeneity (urban, suburban, rural)

### Key Features:
- **Path loss**, **fading**, and **antenna directivity gain**
- **Area-level MIMO channel approximation**
- **Beam-level capacity computation**

---

## 🎲 2. Monte Carlo Channel Modeling

The modeling process includes:

1. **Area Selection**  
   Select M areas out of \(N_{\mathrm{areas}}\) total.

2. **User Generation**  
   For each selected area, generate 500–1000 synthetic user positions inside the polygon boundary.

3. **Environmental Labeling**  
   - Each user is labeled as **indoor** or **outdoor** based on area-specific probabilities.  
   - Each user link is classified as **LoS** or **NLoS** based on elevation angle (3GPP tables).

4. **Channel Coefficients**  
   For every user:
   - Compute **path loss**, **small-scale fading**, and **antenna gain**
   - Output a vector of channel coefficients \( h_{u,s} \)

5. **Area-Level Aggregation**  
   Average all user channels to obtain one representative matrix \( H_m \) for area \( m \).

6. **Capacity Calculation**  
   Using beamforming and spatial multiplexing, compute the capacity per beam from \( H_m \).

---

## 🧠 3. Metaheuristic Optimization for HAPS Placement

In addition to channel simulation, this framework supports **metaheuristic optimization** to solve the **HAPS positioning problem**.  
The objective is to **maximize the sum-rate** over all covered areas by adjusting the HAPS position.

### 🚀 Implemented Algorithms:

- **Particle Swarm Optimization (PSO)**
  - Global Best PSO
  - Local Best PSO
- **Genetic Algorithm (GA)**
- **Differential Evolution (DE)**
- **Custom PSO Variants**
  - `Leader PSO`: particles guided by top-performing swarm members
  - `Democratic PSO`: particles influenced by a consensus of peers
  - `Differential PSO`: PSO with DE-style mutation
- **Meta-PSO with Races**
  - Swarm is partitioned into geographic sub-swarms (“races”)
  - Local race leaders are used to guide updates (e.g., `DLM-PSO`, `ALM-PSO`)

Each algorithm calls the channel model as a black-box objective function and searches the HAPS (x, y) position space to find the location that yields **maximum total capacity**.

### 📈 Outputs:

- Convergence curves
- Comparative performance of all algorithms
- Best-found HAPS coordinates per strategy

---

## 🧪 4. Example Usage

### ✅ Step 1 — Install Requirements

```bash
pip install -r requirements.txt
```

### ✅ Step 2 — Open the Notebook

```bash
jupyter notebook channel_model_demo.ipynb
```

### ✅ Step 3 — Run the Pipeline

In the notebook:

1. Load area maps and user parameters.
2. Generate random users using Monte Carlo simulation.
3. Compute beam-specific capacities using the MIMO channel model.
4. Run one or more metaheuristic algorithms to optimize HAPS location.
5. Visualize convergence and results.

---

## 📊 5. Visualization

After each optimization run, the notebook will:

- Plot convergence history (objective value vs. iteration)
- Compare different metaheuristics on the same scenario
- Report best objective (sum-rate) and best-found HAPS position

This enables a clear evaluation of how each algorithm performs in terms of **efficiency**, **speed**, and **solution quality**.

---

## 📚 6. References

1. **3GPP TR 38.811** — Path loss and LoS/NLoS modeling for non-terrestrial networks  
2. **3GPP TR 38.901** — Channel model for 0.5–100 GHz  
3. **Björnson et al.**, “Massive MIMO networks: Spectral, energy, and hardware efficiency,” *FnT Signal Processing*, 2017  
4. **Takahashi et al.**, “Adaptive power resource allocation with multi-beam directivity control,” *IEEE WCL*, 2019

---

## 🤝 7. Contributing

Want to add a new algorithm? Improve the beamforming logic? Suggest a visualization?  
We welcome contributions! Open an issue or submit a pull request.

---

## 🎉 8. Acknowledgements

This project combines **communication theory** and **intelligent optimization** to explore how HAPS systems can be more effective, efficient, and adaptive in next-generation networks.

Built with ❤️ for research and experimentation.
