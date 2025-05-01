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

## 🌐 1. Channel Model Overview

In each time slot, the HAPS covers a subset of ground **areas**, each potentially containing many users.  
Rather than track each user’s channel in real-time, we approximate each **area**’s effective channel via **Monte Carlo sampling**.  
This allows us to capture user-level fluctuations (e.g., *some are indoors, others outdoors, some enjoy LoS, others Non-LoS*) and then **average** their channel metrics to form a single representative channel per area.

For the equations governing *path loss*, *directivity gain*, and *small-scale fading*, please see:

- **Path Loss** – Refer to [1] and see the implementation in [`env/path_loss.py`](env/path_loss.py).
- **Small-Scale Fading** – Refer to [2] and [3], and see code in [`env/small_scale_fading.py`](env/small_scale_fading.py).
- **Antenna Directivity Gain** – Refer to [4] and the related code in [`env/Directivity_Gain.py`](env/Directivity_Gain.py).

---


## 2. **Monte Carlo Idea**

1. **Area Selection**  
   At each time step, we identify which areas (out of $N_{\mathrm{areas}}$ total) will be served by the HAPS.  
   Suppose we choose $M$ such areas.

2. **Random User Locations**  
   For each chosen area, we generate a **large number** of *synthetic user positions* (e.g., **500 to 1000**) scattered within that area’s boundary.  
   This captures variability in distance, elevation angle, and building penetration conditions.

3. **Environmental Factors**  
   - **Indoor vs. Outdoor**:  
     Each synthetic user is labeled **indoor** or **outdoor** with a probability specific to that area.  
     Example: A **suburban** area might have a lower indoor probability than an **airport** or **industrial** complex.  
   - **LoS vs. Non-LoS**:  
     We assign each user’s link to *LoS* or *Non-LoS* based on the area’s elevation angle distribution or empirical tables (3GPP, etc.).

4. **Channel Computation**  
   - For each synthetic user, we compute their **path loss** (including building entry if indoors), **small-scale fading** (Rician or Rayleigh), and **directivity gain** from the phased array.  
   - This yields an individual *channel coefficient* $h_{u,s}$ for user $u$ in that area and antenna element $s$.

5. **Averaging per Area**  
   Once **all synthetic users** in area $m$ have their channel coefficients, we **average** these into a single “representative” channel matrix $H_m$.  
   Hence, we no longer track each user individually at runtime — instead, we capture the **collective** effect of users in that area.

6. **Beamforming & Capacity**  
   Each area is served by a **dedicated beam**, and capacity is estimated using these averaged channel coefficients.  
   This *simplifies* the problem, yet reflects realistic user diversity, thanks to the Monte Carlo sampling.


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

