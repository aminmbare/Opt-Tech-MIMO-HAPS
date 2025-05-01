import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import plotly.graph_objects as go
import os
import random


class MU_MIMO_CapacityCalculator:
    """
    Class that calculates the capacity for a Multi-User (MU) MIMO system.
    
    This class combines:
    - Small-scale fading (e.g., Rayleigh/Rician).
    - Path loss calculations (LOS, NLOS, building penetration, etc.).
    - Antenna directivity gain.
    
    It computes an averaged channel representation for each cluster (area) 
    and performs MMSE beamforming to estimate the capacity or visualize the 
    beamforming patterns and SINR distributions.
    """
    
    def __init__(self,
                 small_scale_fading,
                 path_loss_calculator,
                 directivity_gain,
                 noise_power_density: float, 
                 bandwidth: float, 
                 N_x: int,
                 N_y: int, 
                 Haps_bounds: List[float]):
        """
        Initializes the MU-MIMO capacity calculator with the necessary components.
        
        Parameters
        ----------
        small_scale_fading : SmallScaleFading
            Instance to generate small-scale fading components/matrices.
        path_loss_calculator : PathLossCalculator
            Instance for path loss calculations (LOS, NLOS, building penetration, etc.).
        directivity_gain : DirectivityGainCalculator
            Instance for calculating directivity gains toward each user/beam.
        noise_power_density : float
            Noise power density in dB or linear scale (W/Hz); interpret as needed.
        bandwidth : float
            System bandwidth in Hz.
        N_x : int
            Number of antennas along the x-axis in the planar array.
        N_y : int
            Number of antennas along the y-axis in the planar array.
        Haps_bounds : list[float]
            Map boundary [min_x, min_y, max_x, max_y] for the HAPS environment.
        """
        self.small_scale_fading = small_scale_fading
        self.path_loss_calculator = path_loss_calculator
        self.directivity_gain = directivity_gain
        self.noise_power_density = noise_power_density
        self.bandwidth = bandwidth
        
        # Example power/transmit parameters
        self.P_tx = 33  # (dBm or dB scale) Transmit power
        self.G_rx = 0   # (dBi) Optional receive antenna gain
        self.G_tx = 43  # (dBi) Optional transmit antenna gain
        
        self.N_x = N_x
        self.N_y = N_y
        self.N_total = N_x * N_y
        self.haps_bounds = Haps_bounds
        
    def set_SINR_map(self, grid_size: int = 150) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create arrays for computing or visualizing the SINR over a 2D grid 
        based on the HAPS bounds.

        Parameters
        ----------
        grid_size : int, optional
            Number of grid points along each axis. Default is 150.

        Returns
        -------
        SINR_map : np.ndarray
            Zero-initialized 2D array for SINR values, shape = (grid_size, grid_size).
        X : np.ndarray
            X-coordinates meshgrid, shape = (grid_size, grid_size).
        Y : np.ndarray
            Y-coordinates meshgrid, shape = (grid_size, grid_size).
        """
        min_x, min_y, max_x, max_y = self.haps_bounds  # Extract map bounds

        # Define the 2D grid within the bounds
        x_range = np.linspace(min_x, max_x, grid_size)
        y_range = np.linspace(min_y, max_y, grid_size)
        
        X, Y = np.meshgrid(x_range, y_range)
        SINR_map = np.zeros_like(X)
        return SINR_map, X, Y
        
    def RSPR_cluster_paramters(self, cluster: Dict, haps_coordinates: List[float]) -> Tuple[np.ndarray, float, float]:
        """
        Compute the average small-scale fading, path loss, and directivity gain for a cluster.

        This function performs a Monte Carlo iteration over N_users random users within the area
        (user positions) in the cluster polygon, computing path loss, directivity, 
        and small-scale fading sums, and then averaging.

        Parameters
        ----------
        cluster : dict
            Dictionary containing:
              - 'N_users' (int): Number of Monte Carlo samples to run.
              - 'SBAND_data': S-band dataset entry with scenario-specific LOS prob, path-loss parameters, etc.
              - 'area_type': str in {'dense_urban','urban','suburban_rural'}.
              - 'elevation_angle_deg': float, angle from HAPS to area.
              - 'indoor_percentage': float, fraction of indoor users.
              - 'traditional_building_percentage': float, fraction of traditional buildings in the area.
              - 'beam_radius': float, radius of the beam covering this area.
              - 'area_centroid': (float, float, float), centroid location (x,y,z=0).
              - 'boundary': shapely.Polygon representing the area boundary.
            Additional keys may be present.
        haps_coordinates : list[float]
            HAPS coordinates in the format [x, y, z].

        Returns
        -------
        small_scale_fading : np.ndarray
            Averaged complex fading vector (length = N_total antennas).
        path_loss : float
            Mean path loss (in dB or linear, as used by the code).
        directivity_gain : float
            Mean directivity gain in dB or linear scale, based on usage.
        """
        N_users = cluster['N_users']
        SBAND_data = cluster['SBAND_data']
        area_type = cluster['area_type']
        if area_type == 'dense_urban':
            scenario_data = SBAND_data.dense_urban
        elif area_type == 'urban':
            scenario_data = SBAND_data.urban
        else:
            scenario_data = SBAND_data.suburban_rural
            
        los_probability = scenario_data.los_probability  # e.g. from dataset
        path_loss_parameters = scenario_data.path_loss
        elevation_angle_deg = cluster['elevation_angle_deg']
        
        total_path_loss = 0.0
        total_directivity_gain = 0.0
        
        indoor_percentage = cluster['indoor_percentage']
        traditional_building_percentage = cluster['traditional_building_percentage']
        beam_radius = cluster['beam_radius']
        beam_centroid = cluster['area_centroid']
        
        total_small_scale_fading = np.zeros(self.N_total, dtype=complex)
        
        for _ in range(N_users):
            # Decide if user is indoor or not
            if np.random.rand() < indoor_percentage:
                los = False
            else:
                # Now decide LOS vs. NLOS
                if np.random.rand() < (los_probability / 100.0):
                    los = True
                else:
                    los = False

            # Generate a random user coordinate inside the polygon
            #while True:
            #    polygon = cluster['boundary']  # Shapely Polygon
            #    min_x, min_y, max_x, max_y = polygon.bounds
            #    rand_x = random.uniform(min_x, max_x)
            #    rand_y = random.uniform(min_y, max_y)
            #    user_point = Point(rand_x, rand_y)
            #    if polygon.contains(user_point):
            #        user_coordinate = [user_point.x, user_point.y, 0.0]
            #        break  # Valid point found

            # Calculate directivity gain for this user
            #total_directivity_gain += self.directivity_gain.calculate_directivity_gain(user_coordinate,
            #                                                                           beam_centroid,
            #                                                                           beam_radius)
            # Calculate path loss
            total_path_loss += self.path_loss_calculator.calculate_path_loss(beam_centroid,
                                                                             haps_coordinates,
                                                                             elevation_angle_deg,
                                                                             los,
                                                                             path_loss_parameters,
                                                                             indoor_percentage,
                                                                             traditional_building_percentage)
            # Sum small-scale fading
            total_small_scale_fading += self.small_scale_fading.calculate_total_H(haps_coordinates,
                                                                                  beam_centroid,
                                                                                  los)
            
        # Averages
        small_scale_fading = total_small_scale_fading / N_users
        path_loss = total_path_loss / N_users
        #directivity_gain = total_directivity_gain / N_users
        
        return small_scale_fading, path_loss, _
    
    def generate_capacities(self,
                            clusters_list: List[Dict],
                            haps_coordinates: List[float],
                            Nr: int,
                            sigma_squared: float = 1e-2,
                            plot: bool = False,
                            map_boundary_points: List[float] = None) -> List[float]:
        """
        Generate the capacity for each cluster (area) using MMSE beamforming.
        
        Parameters
        ----------
        clusters_list : list of dict
            A list of cluster dictionaries, each containing:
              - 'N_users' (int): number of Monte Carlo samples
              - 'SBAND_data': S-band data object
              - 'area_type': str in {dense_urban, urban, suburban_rural}
              - 'boundary': polygon of the area
              - 'indoor_percentage': float
              - 'traditional_building_percentage': float
              - 'beam_radius': float
              - 'area_centroid': [x, y, 0]
              etc.
        haps_coordinates : list[float]
            [x, y, z] of HAPS location
        Nr : int
            Number of clusters/areas to serve (dimension of H_matrix).
        sigma_squared : float, optional
            Regularization / noise term for the beamforming matrix. Default=1e-2.
        plot : bool, optional
            If True, generate optional 3D beam visualization.
        map_boundary_points : list[float], optional
            Unused in this snippet, but could be used for advanced mapping or plotting.

        Returns
        -------
        capacities : list of float
            Capacity in bits/second (or appropriate units) for each of the Nr clusters.
        """
        # Build the channel matrix H_matrix for the Nr clusters
        H_matrix = np.zeros((Nr, self.N_total), dtype=complex)
   
        # Compute path loss, directivity, and fading for each cluster
        for i, cluster in enumerate(clusters_list):
            small_scale_fading, path_loss, _ = self.RSPR_cluster_paramters(cluster, haps_coordinates)
            # Combine all components => amplitude factor from directivity and path loss
            amplitude_factor = np.sqrt(10 ** ((self.G_tx - path_loss) / 10))
            H_matrix[i] = amplitude_factor * small_scale_fading
      
        # MMSE beamforming matrix
        # H_matrix shape: (Nr, N_total)
        # W_mmse shape: (N_total, Nr)
        # Build (H^H * H + sigma^2 I) ^ -1 * H^H
        W_mmse = np.linalg.inv(H_matrix.conj().T @ H_matrix + sigma_squared * np.eye(H_matrix.shape[1])) @ H_matrix.conj().T

        # Normalize columns of W_mmse
        for i_col in range(W_mmse.shape[1]):
            W_mmse[:, i_col] /= np.linalg.norm(W_mmse[:, i_col])
        
        # Calculate capacities
        capacities = []
        for cluster_id in range(H_matrix.shape[0]):
            h_i = H_matrix[cluster_id]         # Channel row for cluster i
            w_i = W_mmse[:, cluster_id]        # Beamforming vector for cluster i
            
            # Signal power
            signal_power = np.abs(np.dot(h_i, w_i))**2 * 10 ** (self.P_tx / 10)
            
            # Interference from other beams
            interference_power = sum(
                np.abs(np.dot(h_i, W_mmse[:, j]))**2 * 10 ** (self.P_tx / 10)
                for j in range(H_matrix.shape[0]) if j != cluster_id
            )
            
            # Noise power
            noise_power = sigma_squared * self.bandwidth
            # Effective SINR
            SINR_i = signal_power / (interference_power + 10 ** (self.noise_power_density / 10))

            # Shannon capacity
            capacity_i = self.bandwidth * np.log2(1 + SINR_i)
            capacities.append(capacity_i)
        
        # Optional plotting
        if plot:
            self.visualize_beamforming_patterns_superimposed_3D_plotly(W_mmse,
                                                                      self.N_x,
                                                                      self.N_y,
                                                                      clusters_list)

        return capacities


    def compute_SINR_map_and_plot(self,
                                  W_mmse: np.ndarray,
                                  haps_coordinates: List[float],
                                  clusters_list: List[Dict],
                                  grid_size: int = 150):
        """
        Compute and visualize the SINR level over a 2D grid, considering each cluster's boundary.

        Parameters
        ----------
        W_mmse : np.ndarray
            MMSE beamforming weights (shape: (N_total, N_clusters)).
        haps_coordinates : list[float]
            [x, y, z] coordinates of the HAPS.
        clusters_list : list of dict
            Each dict containing 'area_name' and 'boundary' (Shapely Polygon),
            plus other cluster info for path loss and small-scale fading.
        grid_size : int, optional
            Number of grid points per axis in the map. Default=150.
        """
        print("computing maps interference")
        N_clusters = len(clusters_list)
        SINR_map, X, Y = self.set_SINR_map(grid_size)
        
        # Loop over grid points
        for i in range(grid_size):
            for j in range(grid_size):
                grid_x = X[i, j]
                grid_y = Y[i, j]
                tile_point = Point(grid_x, grid_y)

                assigned_beam = None
                # Identify which cluster boundary covers this grid point
                for cluster_id, cluster in enumerate(clusters_list):
                    boundary = cluster['boundary']
                    if boundary.contains(tile_point):
                        assigned_beam = cluster_id
                        break

                if assigned_beam is None:
                    # This grid point not covered by any cluster
                    continue

                # Monte Carlo for user at tile location
                cluster = clusters_list[assigned_beam]
                cluster["N_users"] = 100  # e.g. sample 100 times
                cluster["user_coordinate"] = [grid_x, grid_y, 0]

                # Compute path loss / directivity / fading
                small_scale_fading, path_loss, directivity_gain = self.RSPR_cluster_paramters(cluster, haps_coordinates)
                
                # Combine these into channel vector
                h_tile = np.sqrt(10 ** ((directivity_gain - path_loss) / 10)) * small_scale_fading
                
                # Signal power
                signal_power = np.abs(np.dot(h_tile, W_mmse[:, assigned_beam]))**2 * 10 ** (self.P_tx / 10)
                
                # Interference from all beams except assigned
                interference_power = sum(
                    np.abs(np.dot(h_tile, W_mmse[:, k]))**2 * 10 ** (self.P_tx / 10)
                    for k in range(N_clusters) if k != assigned_beam
                )
                
                SINR_i = signal_power / (interference_power + 10 ** (self.noise_power_density / 10))
                SINR_map[i, j] += SINR_i

        # Replace zeros or negative values with NaN
        SINR_map_for_plot = np.where(SINR_map > 0, SINR_map, np.nan)

        # Plot the resulting SINR map
        plt.figure(figsize=(10, 8))
        plt.contourf(X, Y, 10 * np.log10(SINR_map_for_plot), levels=50, cmap='viridis')

        # Plot HAPS position
        haps_x, haps_y = haps_coordinates[0], haps_coordinates[1]
        plt.scatter(haps_x, haps_y, color='red', label='HAPS Position', s=100, marker='x')

        plt.colorbar(label='Interference Power (dB)')
        plt.title("Interference Heatmap Within Map and Area Boundaries")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")

        # Overlay area boundaries
        for cluster in clusters_list:
            boundary = cluster['boundary']
            if hasattr(boundary, 'exterior'):  # For polygons
                area_x, area_y = boundary.exterior.xy
                plt.plot(area_x, area_y, linewidth=2, label=cluster['area_name'])

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.close()

    @staticmethod
    def visualize_beamforming_pattern_planar(W_mmse: np.ndarray,
                                             N_x: int,
                                             N_y: int,
                                             cluster_list: List[Dict],
                                             N_phi: int = 360,
                                             N_theta: int = 90):
        """
        Visualize a planar beamforming pattern (2D) at a fixed elevation angle.
        
        Parameters
        ----------
        W_mmse : np.ndarray
            Beamforming weights, shape (N_antennas, N_clusters).
        N_x : int
            Number of antennas along x-axis.
        N_y : int
            Number of antennas along y-axis.
        cluster_list : list
            Clusters, each with at least 'area_name'.
        N_phi : int
            Number of azimuth angle samples. Default=360.
        N_theta : int
            Not actively used here (only a single fixed angle).
        """
        N_antennas, N_clusters = W_mmse.shape
        assert N_antennas == N_x * N_y, f"Mismatch: Antennas {N_antennas} != {N_x * N_y}"

        # Define azimuth angles
        phi = np.linspace(0, 2 * np.pi, N_phi) - np.pi  # [-180..180] range
        theta_fixed = np.pi/4  # e.g., 45 deg

        def steering_vector_planar_fixed_elevation(ph: float, Nx: int, Ny: int, th: float = np.pi/6) -> np.ndarray:
            """
            Generate a flattened steering vector for a planar array 
            at fixed elevation 'th' and azimuth 'ph'.
            """
            n_x = np.arange(Nx)[:, None]
            n_y = np.arange(Ny)[None, :]
            phase_x = np.exp(1j * np.pi * n_x * np.sin(th) * np.cos(ph))
            phase_y = np.exp(1j * np.pi * n_y * np.sin(th) * np.sin(ph))
            return (phase_x * phase_y).flatten()

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        
        for cluster_id in range(N_clusters):
            area_names = cluster_list[cluster_id]['area_name']
            W_cluster = W_mmse[:, cluster_id]
            gain = np.zeros(N_phi)
            
            # Compute gain for each azimuth angle
            for j, p in enumerate(phi):
                a_phi = steering_vector_planar_fixed_elevation(p, N_x, N_y, theta_fixed)
                gain[j] = np.abs(W_cluster.conj().T @ a_phi)**2
            
            # Normalize
            gain /= np.max(gain)
            
            ax.plot(phi, gain, label=f"{area_names}")

        ax.set_title("Beamforming Patterns (2D, fixed elevation=45°)")
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
        plt.show()
    
    @staticmethod
    def visualize_beamforming_patterns_superimposed_3D(W_mmse: np.ndarray,
                                                       N_x: int,
                                                       N_y: int,
                                                       cluster_list: List[Dict],
                                                       N_phi: int = 360,
                                                       N_theta: int = 90):
        """
        Visualize superimposed 3D beamforming patterns for a planar array (matplotlib).

        Parameters
        ----------
        W_mmse : np.ndarray
            Beamforming weights, shape (N_antennas, N_clusters).
        N_x : int
            Number of antennas along x-axis.
        N_y : int
            Number of antennas along y-axis.
        cluster_list : list
            List of clusters, each with 'area_name'.
        N_phi : int
            Number of azimuth angles.
        N_theta : int
            Number of elevation angles.
        """
        N_antennas, N_clusters = W_mmse.shape
        assert N_antennas == N_x * N_y, f"Mismatch: Antennas {N_antennas} != {N_x * N_y}"

        phi = np.linspace(-np.pi, np.pi, N_phi) 
        theta = np.linspace(0, np.pi / 2, N_theta)

        def steering_vector_planar(th: float, ph: float, Nx: int, Ny: int) -> np.ndarray:
            """Generate the flattened steering vector for a planar array at (theta, phi)."""
            n_x = np.arange(Nx)[:, None]
            n_y = np.arange(Ny)[None, :]
            phase_x = np.exp(1j * np.pi * n_x * np.sin(th) * np.cos(ph))
            phase_y = np.exp(1j * np.pi * n_y * np.sin(th) * np.sin(ph))
            return (phase_x * phase_y).flatten()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']

        for cluster_id in range(N_clusters):
            area_names = cluster_list[cluster_id]['area_name']
            W_cluster = W_mmse[:, cluster_id]

            gain = np.zeros((N_theta, N_phi))
            for i, t in enumerate(theta):
                for j, p in enumerate(phi):
                    a_theta_phi = steering_vector_planar(t, p, N_x, N_y)
                    gain[i, j] = np.abs(W_cluster.conj().T @ a_theta_phi)**2

            # Convert (theta, phi) + gain => Cartesian
            azimuth, elevation = np.meshgrid(phi, theta)
            x = gain * np.sin(elevation) * np.cos(azimuth)
            y = gain * np.sin(elevation) * np.sin(azimuth)
            z = gain * np.cos(elevation)
            
            ax.plot_surface(x, y, z, 
                            color=colors[cluster_id % len(colors)], 
                            alpha=0.6,
                            label=area_names,
                            edgecolor='none')

        ax.set_title("Superimposed 3D Beamforming Patterns")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Gain (dBi)")
        plt.show()

    @staticmethod
    def visualize_beamforming_patterns_superimposed_3D_plotly(W_mmse: np.ndarray,
                                                             N_x: int,
                                                             N_y: int,
                                                             cluster_list: List[Dict],
                                                             N_phi: int = 360,
                                                             N_theta: int = 90):
        """
        Visualize superimposed 3D beamforming gains (Plotly),
        each beam in a single solid color, with a matching legend entry.

        Parameters
        ----------
        W_mmse : np.ndarray
            Beamforming weights, shape (N_antennas, N_clusters).
        N_x : int
            Number of antennas along the x-axis.
        N_y : int
            Number of antennas along the y-axis.
        cluster_list : list of dict
            Each dict should have 'area_name' at least.
        N_phi : int, optional
            Number of azimuth angles (default=360).
        N_theta : int, optional
            Number of elevation angles (default=90).
        """
        N_antennas, N_clusters = W_mmse.shape
        assert N_antennas == N_x * N_y, (
            f"Mismatch: Antennas {N_antennas} != {N_x * N_y}"
        )

        phi = np.linspace(-np.pi, np.pi, N_phi)
        theta = np.linspace(0, np.pi / 2, N_theta)

        def steering_vector_planar(th: float, ph: float, Nx: int, Ny: int) -> np.ndarray:
            """Generate the flattened steering vector for a planar array at (theta=th, phi=ph)."""
            n_x = np.arange(Nx)[:, None]
            n_y = np.arange(Ny)[None, :]
            phase_x = np.exp(1j * np.pi * n_x * np.sin(th) * np.cos(ph))
            phase_y = np.exp(1j * np.pi * n_y * np.sin(th) * np.sin(ph))
            return (phase_x * phase_y).flatten()

        fig = go.Figure()
        azimuth, elevation = np.meshgrid(phi, theta)

        beam_colors = ["blue", "orange", "green", "red", "purple",
                       "brown", "pink", "gray", "cyan", "magenta"]

        for cluster_id in range(N_clusters):
            area_name = cluster_list[cluster_id]["area_name"]
            W_cluster = W_mmse[:, cluster_id]

            gain = np.zeros((N_theta, N_phi))
            for i, t in enumerate(theta):
                for j, p in enumerate(phi):
                    a_tp = steering_vector_planar(t, p, N_x, N_y)
                    gain[i, j] = np.abs(W_cluster.conj().T @ a_tp) ** 2

            x = gain * np.sin(elevation) * np.cos(azimuth)
            y = gain * np.sin(elevation) * np.sin(azimuth)
            z = gain * np.cos(elevation)

            color = beam_colors[cluster_id % len(beam_colors)]
            solid_color_scale = [[0, color], [1, color]]

            fig.add_trace(go.Surface(
                x=x, y=y, z=z,
                surfacecolor=np.ones_like(gain),  # uniform color
                colorscale=solid_color_scale,
                cmin=0, cmax=1,
                showscale=False,
                opacity=0.8,
                name=area_name,
                showlegend=False,
                legendgroup=area_name
            ))

            # Dummy scatter for the legend color
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='markers',
                marker=dict(size=8, color=color),
                name=area_name,
                legendgroup=area_name,
                showlegend=True
            ))

        fig.update_layout(
            title="Superimposed 3D Beamforming Patterns",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Gain"
            ),
            width=900,
            height=700,
            legend=dict(
                x=1.05,
                y=1.0
            )
        )

        fig.show()
