import numpy as np



## references 
## E. Björnson, J. Hoydis, and L. Sanguinetti,  “Massive mimo networks: Spectral, energy, and hardware efficiency,”  *Foundations and Trends® in Signal Processing*, vol. 11, pp. 154–655, 01 2017.
## 3GPP, “Study on channel model for frequencies from 0.5 to 100 GHz,” 3rd Generation Partnership Project (3GPP), Technical Report (TR) 38.901,  05 2017, version 14.0.0


class SmallScaleFading:
    """
    Class for modeling small-scale fading (LOS and NLOS) in a MIMO channel between
    a transmitter (e.g., HAPS) and a receiver (e.g., ground cluster).
    
    This includes:
    - Rician/LOS paths (with user-defined K-factor).
    - Rayleigh/NLOS components.
    - Computation of array responses for planar or linear arrays.
    
    Attributes
    ----------
    Nt : int
        Number of transmit antennas (e.g., on HAPS).
    Nr : int
        Number of receive antennas (e.g., on a ground cluster).
    wavelength : float
        Wavelength of the carrier signal in meters.
    K_factor : float
        Rician K-factor specifying LOS to NLOS power ratio.
    d_tx : float
        Spacing between transmit antennas (default ~ half-wavelength).
    d_rx : float
        Spacing between receive antennas (default ~ half-wavelength).
    """
    
    def __init__(self, 
                 Nt: int, 
                 Nr: int, 
                 wavelength: float, 
                 K_factor: float = 5, 
                 d_tx: float = 0.075, 
                 d_rx: float = 0.075):
        """
        Initialize the small-scale fading model.

        Parameters
        ----------
        Nt : int
            Number of transmit antennas.
        Nr : int
            Number of receive antennas.
        wavelength : float
            Wavelength of the signal in meters.
        K_factor : float, optional
            Rician factor (default=5), representing LOS/NLOS ratio.
        d_tx : float, optional
            Spacing between transmit antennas (default=0.075 m, approx half-wavelength at 2 GHz).
        d_rx : float, optional
            Spacing between receive antennas (default=0.075 m).
        """
        self.Nt = Nt
        self.Nr = Nr
        self.wavelength = wavelength
        self.K_factor = K_factor
        self.d_tx = d_tx
        self.d_rx = d_rx

    def calculate_angles(self, tx_pos: tuple, rx_pos: tuple) -> tuple:
        """
        Calculate azimuth and elevation angles for both AoD (Angle of Departure)
        and AoA (Angle of Arrival), assuming a direct line from tx to rx.

        Parameters
        ----------
        tx_pos : tuple (float, float, float)
            Coordinates of the transmitter (x, y, z).
        rx_pos : tuple (float, float, float)
            Coordinates of the receiver (x, y, z).

        Returns
        -------
        (azimuth_aod, elevation_aod, azimuth_aoa, elevation_aoa) : tuple of floats
            Azimuth and elevation angles (in radians) for both AoD and AoA.
            In this simplified model, AoD = AoA for both azimuth and elevation.
        """
        d_horizontal = np.sqrt((rx_pos[0] - tx_pos[0]) ** 2 + (rx_pos[1] - tx_pos[1]) ** 2)
        height_diff = tx_pos[2] - rx_pos[2]

        # Azimuth angles (AoD = AoA)
        azimuth_aod = np.arctan2(rx_pos[1] - tx_pos[1], rx_pos[0] - tx_pos[0])
        azimuth_aoa = azimuth_aod

        # Elevation angles (AoD = AoA)
        elevation_aod = np.arctan2(d_horizontal, height_diff)
        elevation_aoa = elevation_aod

        return azimuth_aod, elevation_aod, azimuth_aoa, elevation_aoa

    def array_response(self, 
                       N: int, 
                       d: float, 
                       wavelength: float, 
                       azimuth: float, 
                       elevation: float) -> np.ndarray:
        """
        Compute the array response vector for a linear array of size N.

        Parameters
        ----------
        N : int
            Number of antennas in the array.
        d : float
            Spacing between adjacent antennas (in meters).
        wavelength : float
            Wavelength of the signal (in meters).
        azimuth : float
            Azimuth angle (radians).
        elevation : float
            Elevation angle (radians).

        Returns
        -------
        response : np.ndarray (complex)
            Array response vector of length N.
        """
        k = 2 * np.pi / wavelength
        # Phase shift increments across array elements
        response = np.exp(1j * k * d * np.arange(N) * np.sin(elevation) * np.cos(azimuth))
        return response

    def array_response_grid(self, 
                            N: int, 
                            d: float, 
                            wavelength: float, 
                            azimuth: float, 
                            elevation: float) -> np.ndarray:
        """
        Compute the array response for a 2D (MxN) grid, flattened into a single vector.
        This can approximate a planar array rather than just a linear array.

        Parameters
        ----------
        N : int
            Currently unused to define M or N dimension. (Kept for consistency.)
        d : float
            Antenna spacing.
        wavelength : float
            Signal wavelength.
        azimuth : float
            Azimuth angle in radians.
        elevation : float
            Elevation angle in radians.

        Returns
        -------
        np.ndarray (complex)
            Flattened array response of shape (M*N,).
            Example uses M=N=4, but can be adapted if needed.
        """
        k = 2 * np.pi / wavelength
        # For demonstration, define M=N=4 for a 4x4 planar array
        M, Nx = 4, 4
        d_x, d_y = d, d
        response_2d = np.zeros((M, Nx), dtype=complex)

        for m in range(M):
            for n in range(Nx):
                phase_shift = k * (m * d_x * np.sin(elevation) * np.cos(azimuth) +
                                   n * d_y * np.sin(elevation) * np.sin(azimuth))
                response_2d[m, n] = np.exp(1j * phase_shift)

        # Flatten the 2D array response
        return response_2d.flatten()

    def calculate_H_LOS(self, 
                        tx_pos: tuple, 
                        rx_pos: tuple) -> np.ndarray:
        """
        Compute the LOS (Line-of-Sight) component of the channel matrix (Nr x Nt).

        Parameters
        ----------
        tx_pos : tuple (float, float, float)
            Transmitter coordinates (x, y, z).
        rx_pos : tuple (float, float, float)
            Receiver coordinates (x, y, z).

        Returns
        -------
        H_LOS : np.ndarray (complex)
            LOS channel matrix of shape (Nr, Nt).
        """
        azimuth_aod, elevation_aod, azimuth_aoa, elevation_aoa = self.calculate_angles(tx_pos, rx_pos)

        # Planar array at Tx, linear array at Rx (or vice versa).
        a_tx = self.array_response_grid(self.Nt, self.d_tx, self.wavelength, azimuth_aod, elevation_aod)
        a_rx = self.array_response(self.Nr, self.d_rx, self.wavelength, azimuth_aoa, elevation_aoa)

        # Distance-based phase factor
        distance_3d = np.sqrt((rx_pos[0] - tx_pos[0])**2 +
                              (rx_pos[1] - tx_pos[1])**2 +
                              (rx_pos[2] - tx_pos[2])**2)
        phase_factor = np.exp(-1j * 2 * np.pi * distance_3d / self.wavelength)

        # Outer product to form LOS channel: (Nr x Nt)
        H_LOS = np.outer(a_rx, a_tx.conj()) * phase_factor
        return H_LOS

    def calculate_H_NLOS(self) -> np.ndarray:
        """
        Compute the NLOS (Rayleigh) fading component of the channel matrix (Nr x Nt).

        Returns
        -------
        H_NLOS : np.ndarray (complex)
            Random NLOS channel matrix with shape (Nr, Nt).
        """
        real_part = np.random.normal(0, 1, (self.Nr, self.Nt))
        imag_part = np.random.normal(0, 1, (self.Nr, self.Nt))
        H_NLOS = (real_part + 1j * imag_part) / np.sqrt(2)
        return H_NLOS

    def calculate_total_H(self, 
                          tx_pos: tuple, 
                          rx_pos: tuple, 
                          LOS_condition: bool) -> np.ndarray:
        """
        Compute the total channel (LOS + NLOS) in a Rician manner if LOS_condition is True,
        or purely NLOS (Rayleigh) if LOS_condition is False.

        Parameters
        ----------
        tx_pos : tuple (float, float, float)
            Transmitter coordinates.
        rx_pos : tuple (float, float, float)
            Receiver coordinates.
        LOS_condition : bool
            True if link is considered LOS; otherwise NLOS.

        Returns
        -------
        H : np.ndarray (complex)
            Flattened channel matrix of size (Nr * Nt,).
            The caller can reshape it if needed.
        """
        H_NLOS = self.calculate_H_NLOS()  # shape (Nr, Nt)

        if LOS_condition:
            H_LOS = self.calculate_H_LOS(tx_pos, rx_pos)  # shape (Nr, Nt)
            # Rician combination
            alpha_los = np.sqrt(self.K_factor / (self.K_factor + 1))
            alpha_nlos = np.sqrt(1 / (self.K_factor + 1))
            H = alpha_los * H_LOS + alpha_nlos * H_NLOS
            return H.reshape(-1)  # Flatten
        else:
            # Pure Rayleigh (NLOS)
            return H_NLOS.reshape(-1)
