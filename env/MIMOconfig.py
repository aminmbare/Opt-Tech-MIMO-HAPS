from dataclasses import dataclass




@dataclass
class MIMOConfig:
    ## Number of antennas in the x axis 
    N_x : int = 4
    ## Number of antennas in the y axis
    N_y : int = 4
   ## Transmitted power in dBm
    P_tx : float = 41 
    ## Received Gain in dB
    G_rx : float = 0
    ## Noise Power in dBm
    P_n : float = -100
    ## Bandwidth in MHz
    B : float = 20
    ## Frequency in GHz
    f : float = 2
    ## Distance between antennas in meters
    d_x  : float = 0.15
    d_y  : float = 0.15
    
    ## antenna_aperture_efficiency
    eta : float = 0.95
    
    ## Haps altitude in meters
    
    H : float = 20_000.0
    
    
    
    
## Change the parameters here to change the MIMO configuration
## Example of how to use the MIMOConfig class

MIMOConfig = MIMOConfig(
    N_x=4,
    N_y=4,
    P_tx=41,
    G_rx=0,
    P_n=-100,
    B=20,
    f=2,
    d_x=0.15,
    d_y=0.15
)


   