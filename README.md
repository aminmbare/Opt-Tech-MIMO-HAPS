# **HAPS-Based MIMO Channel Model**

This repository introduces a **comprehensive channel model** for a High Altitude Platform Station (HAPS) that links the platform’s position to **beam-specific capacity**. The model incorporates crucial factors such as **LOS/NLOS conditions**, **area heterogeneity** (dense urban, urban, and rural), and stochastic user distributions (indoor/outdoor). The system leverages a **Multi-User MIMO (MU-MIMO)** approach to serve multiple areas simultaneously via spatial multiplexing, reflecting realistic propagation and user diversity.

> **Folder Structure Highlights:**
> - **`Area Distribution/`**: Contains CSV files (`clustered_areas_r_*.csv`) specifying different area configurations.  
> - **`env/`**: Holds the main channel model components (e.g., `ChannelModel.py`, `Map.py`, `MIMOconfig.py`, etc.).  
> - **`channel_model_demo.ipynb`**: An example notebook demonstrating how to run channel model.  
> - **`requirements.txt`**: A list of the Python dependencies needed to replicate the environment.
---


## 1. **Overview**

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


## 3. **Example Usage**

1. **Install** the required packages. You can use:
   - **pip** with the included `requirements.txt`:
     ```bash
     pip install --upgrade pip
     pip install -r requirements.txt
     ```
   - or **Miniconda** / **Anaconda** to create a virtual environment and install packages (e.g., `conda create --name haps_env python=3.9 && conda activate haps_env && pip install -r requirements.txt`).

2. **Clone** this repository and **launch** the main notebook (`channel_model_demo.ipynb`).

3. **Set** your scenario parameters: number of areas ($M$), indoor probabilities, LoS/NLoS distribution, etc.

4. **Run** the simulation. You’ll see:
   - Maps of area polygons with random user locations  
   - Beamforming gains and capacity results for each selected area


---


## 4. **References**

1. **Path Loss**  
   3GPP, “Study on New Radio (NR) to support non-terrestrial networks,”  
   3rd Generation Partnership Project (3GPP), Technical Report (TR) 38.811,  
   09 2017, version 15.4.0.

2. **Small-Scale Fading** (Massive MIMO overview)  
   E. Björnson, J. Hoydis, and L. Sanguinetti,  
   “Massive mimo networks: Spectral, energy, and hardware efficiency,”  
   *Foundations and Trends® in Signal Processing*, vol. 11, pp. 154–655, 01 2017.

3. **Small-Scale Fading** (3GPP)  
   3GPP, “Study on channel model for frequencies from 0.5 to 100 GHz,”  
   3rd Generation Partnership Project (3GPP), Technical Report (TR) 38.901,  
   05 2017, version 14.0.0.

4. **Directivity Gain**  
   M. Takahashi, Y. Kawamoto, N. Kato, A. Miura, and M. Toyoshima,  
   “Adaptive power resource allocation with multi-beam directivity control  
   in high-throughput satellite communication systems,” *IEEE Wireless  
   Communications Letters*, vol. 8, no. 4, pp. 1248–1251, 2019.
---

**Enjoy exploring** the Monte Carlo–driven HAPS MIMO channel model!  
If you have any questions or contributions, **open an issue** or **submit a pull request**.
