"""Reproduce la tabla de separación sobre sintéticos (Tabla 1 del informe)."""
import numpy as np
from craterscore import morphometric_metrics, synthetic_crater

SEED, WIN = 20260906, (512, 512)
rng = np.random.default_rng(SEED)
yy, xx = np.mgrid[0:WIN[0], 0:WIN[1]].astype(float)
valle = (-80.0 * np.exp(-((xx - 256) ** 2) / (2 * 40.0 ** 2))
         + rng.normal(0, 3, WIN))

casos = {
    "cráter fresco":   synthetic_crater(shape=WIN, r_rim=70, dh_rim=40, dh_floor=120,
                                        sigma_r=17, sigma_f=42, noise_std=2, seed=SEED),
    "cráter degradado": synthetic_crater(shape=WIN, r_rim=80, dh_rim=8, dh_floor=25,
                                         sigma_r=25, sigma_f=50, trend=(.05, -.03),
                                         noise_std=4, seed=SEED),
    "elipse 2.2:1":    synthetic_crater(shape=WIN, r_rim=70, ellipticity=2.2,
                                        noise_std=2, seed=SEED),
    "anticráter":     -synthetic_crater(shape=WIN, r_rim=70, noise_std=2, seed=SEED),
    "ruido blanco":    rng.normal(0, 5, WIN),
    "rampa planar":    synthetic_crater(shape=WIN, dh_rim=0, dh_floor=0,
                                        trend=(.4, .15), noise_std=3, seed=SEED),
    "valle lineal":    valle,
}

hdr = f"{'caso':18s} {'R2':>5s} {'r_rim':>7s} {'coher':>6s} {'anillo':>7s} {'BCI':>5s} {'borde?':>7s}"
print(hdr); print("-" * len(hdr))
for k, d in casos.items():
    m = morphometric_metrics(d)
    print(f"{k:18s} {m['r2']:5.2f} {m['r_rim_px']:7.1f} {m['coherence']:6.2f} "
          f"{m['ring_contrast']:+7.2f} {m['bci']:5.2f} {str(m['edge_limited']):>7s}")
