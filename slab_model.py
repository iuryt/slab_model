import numpy as np
import xarray as xr
from scipy import signal
import gsw


def slab_model(time, wind_stress, H, latitude, 
               Zi0=0, rho0=1025, ni_band=[0.8, 1.2], 
               drag=0.15, wind_work_method="Majumder et al (2015)",
               numerical_method=1
              ):
    """
    Compute mixed-layer ocean currents using a slab model.
    This is based on the MatLab's code provided by Tom Farrar (WHOI).

    Parameters:
    - time: 1-D numpy array, time in seconds.
    - wind_stress: 1-D numpy array, wind stress (taux + 1j * tauy) in Pascal.
    - H: 1-D numpy array, mixed-layer depth in meters.
    - latitude: float, latitude in degrees.
    - Zi0: float, initial condition for mixed-layer currents.
    - rho0: float, ocean density in kg/m^3.
    - ni_band: list, near-inertial band as a fraction of the inertial frequency (f).
    - drag: float, fraction of the inertial frequency (f) representing drag.
    - wind_work_method: str, method for calculating wind work ('Majumder et al (2015)' or 'Plueddemann and Farrar (2006)').
    - numerical_method: int, numerical method for time stepping the model (0 for Forward Euler and 1 for an adapted version of D'Asaro (1985) eq A1 for varying H)

    Returns:
    - output: xarray.Dataset containing computed variables:
        - Zi: Mixed-layer currents.
        - Tw: Wind stress divided by rho0 (m2/s2).
        - Zif: Filtered mixed-layer currents with near-inertial band.
        - Twf: Filtered wind stress with near-inertial band.
        - H: Mixed-layer depth.
        - wind_work: Wind work on mixed-layer currents.
        - near_inertial_filter: Near-inertial band filter.

    Note:
    - wind_stress and H should have the same size as time.
    
    Author:
    iury simoes-sousa (iury@whoi.edu)
    """

    dt = np.diff(time).mean()

    # Planetary vorticity (inertial frequency)
    omega = 2 * np.pi / (23 * 3600 + 56 * 60 + 4.1) # 23 h 56 min 4.1 s
    f = 2 * omega * np.sin(latitude * np.pi / 180)

    ip = (2 * np.pi / np.abs(f))  # Inertial period in seconds

    r = drag * np.abs(f)  # 0.15 As in Alford 2001 (JPO), 2003 (GRL)
    wP = r + 1j * f

    # N/m2 = kg/(m s2)
    Tw = wind_stress / rho0 # kg/(m s2) * m3/kg = m2/s2 
    
    
    Zi = Tw.copy() * 0
    Zi[0] = Zi0
    
    if numerical_method == 0:
        step_forward = lambda i: Zi[i - 1] * np.exp(-wP * dt) + (Tw[i] / H[i]) * dt
    elif numerical_method == 1:
        step_forward = lambda i: (
                Zi[i - 1] * np.exp(-wP * dt)
                - ((Tw[i] - Tw[i - 1]) / H[i] + Tw[i] * (1 / H[i] - 1 / H[i - 1])) / (dt * wP ** 2)
                * (1 - np.exp(-wP * dt))
            )
    else:
        raise ValueError("Invalid numerical method.")
    
    # time-stepping
    for i in range(1, len(Tw)):
        Zi[i] = step_forward(i)
        
    # 4th order Butterworth filter for the near-inertial band
    b, a0 = signal.butter(4, np.array(ni_band) * np.abs(f) / (2 * np.pi),
                          fs=1 / dt, btype="bandpass")
    freq, h = signal.freqz(b, a0, fs=1 / dt, worN=2000)

    bfilter = xr.DataArray(np.abs(h) / np.abs(h).max(), dims=["frequency"], coords=dict(frequency=("frequency", freq)))

    # Filter the wind forcing
    valid_Tw = ~np.isnan(Tw)
    Twf = signal.lfilter(b, a0, signal.detrend(Tw[valid_Tw]))
    Twf = signal.lfilter(b, a0, Twf[valid_Tw][::-1])[::-1]

    # Filter mixed-layer currents
    valid_Zi = ~np.isnan(Zi)
    Zif = Zi.copy() * np.nan
    Zif[valid_Zi] = signal.lfilter(b, a0, signal.detrend(Zi[valid_Zi]))
    Zif[valid_Zi] = signal.lfilter(b, a0, Zif[valid_Zi][::-1])[::-1]

    ones = xr.DataArray(
        np.ones_like(time), dims=["time"],
        coords=dict(time=("time", time))
    )

    if wind_work_method == "Majumder et al (2015)":
        wind_work = ((Zif.conj() * Twf * rho0).real)
    elif wind_work_method == "Plueddemann and Farrar (2006)":
        ekman_currents_dt = xr.DataArray(
            Tw / H, dims=["time"], coords={"time": time}
        ).differentiate("time").values
        wind_work = (-rho0 * H * ((Zif.conj() / (1j * f)) * ekman_currents_dt).real)
    else:
        raise ValueError("Invalid wind work method.")

    output = xr.Dataset(dict(
        Zi=ones * Zi,
        Tw=ones * Tw,
        Zif=ones * Zif,
        Twf=ones * Twf,
        H=ones * H,
        wind_work=ones * wind_work,
        near_inertial_filter=bfilter,
    ))

    output.attrs = dict(ip=ip, drag=drag, f=f, latitude=latitude, dt=dt, ni_band=ni_band)

    return output