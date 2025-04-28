import math


# M. Takahashi, Y. Kawamoto, N. Kato, A. Miura, and M. Toyoshima,  “Adaptive power resource allocation with multi-beam directivity control  in high-throughput satellite communication systems,” *IEEE Wireless    Communications Letters*, vol. 8, no. 4, pp. 1248–1251, 2019.

class DirectivityGainCalculator:
    """
    This class computes the directivity gain  for a user 
    served by a HAPS beam,

    The constructor requires:
    - H (float):    altitude of the HAPS.
    - eta (float):  aperture efficiency.

    The 'calculate_directivity_gain' method takes:
    - user_coord:      (x_u, y_u) ground coordinates of the user
    - beam_centroid:   (x_b, y_b) ground coordinates of the beam center
    - beam_radius:     r (float) radius of the beam in ground projection

    Returns the directivity gain (dB).
    """

    def __init__(self, H, eta):
        """
        Parameters
        ----------
        H : float
            Altitude of the HAPS (same units as the X-Y plane).
        eta : float
            Aperture efficiency (0 < eta <= 1). 
        """
        self.H = H
        self.eta = eta

    def calculate_directivity_gain(self, user_coord, beam_centroid, beam_radius):
        """
        Compute the directivity gain for a user at 'user_coord',
        when the beam center is 'beam_centroid' and the beam has radius 'beam_radius'.
        
        Steps (matching your snippet):
          1) Compute half-power beamwidth:
               theta_3dB = 2 * atan(r / H).
          2) Compute G_m^0 in linear:
               G_m^0 (lin) = eta * ( (70*pi) / theta_3dB )^2.
          3) Convert G_m^0 to dB:
               [G_m^0]_dB = 10*log10(G_m^0).
          4) Compute user elevation angle:
               theta_user = arctan( horizontal_dist / H ).
          5) Compute user directivity [G_user]_dB:
               [G_user]_dB = [G_m^0]_dB 
                             - 12*( G_m^0/eta )*( (theta_user)/(70*pi) )^2
    
        Parameters
        ----------
        user_coord : tuple(float, float)
            (x_u, y_u) for the user ground location.
        beam_centroid : tuple(float, float)
            (x_b, y_b) for the beam center location.
        beam_radius : float
            Radius of the beam on the ground.

        Returns
        -------
        float
            Directivity gain (dB).
        """
        
        x_u, y_u, _ = user_coord
        x_b, y_b, _ = beam_centroid
        
        # 1) Half-power beamwidth (in radians).
        #    This is an approximation: full cone angle that subtends radius r at altitude H.
        theta_3dB = 2.0 * math.atan2(beam_radius, self.H)
        
        # 2) G_m^0 in linear
        #    G_m^0 = eta * ( (70 * pi) / theta_3dB )^2
        G_m0_lin = self.eta * ((70.0 * math.pi) / theta_3dB)**2
        
        # 3) [G_m^0]_dB
        G_m0_dB = 10.0 * math.log10(G_m0_lin)
        
        # 4) Horizontal distance
        dx = x_u - x_b
        dy = y_u - y_b
        horizontal_distance = math.sqrt(dx*dx + dy*dy)
        
        #    User angle: arctan( horizontal / H )
        theta_user = math.atan2(horizontal_distance, self.H)
        
        # 5) [G_user]_dB = [G_m^0]_dB - 12 * (G_m^0 / eta)*((theta_user)/(70*pi))^2
        #    Following your snippet exactly
        loss_term = 12.0 * (G_m0_lin / self.eta) * (theta_user / (70.0*math.pi))**2
        G_user_dB = G_m0_dB - loss_term
        
        
        
        
        return G_user_dB


