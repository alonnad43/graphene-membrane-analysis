# Mathematical Models & Equations for Chemistry Simulation

## Adsorption Kinetics (Pseudo-Second-Order)

$$
\frac{dq}{dt} = k_2 (q_{max} - q)^2
$$

Python:
```python
def pseudo_second_order_adsorption(q, qmax, k2):
    return k2 * (qmax - q)**2
```

## Bacterial Inactivation (First-Order/Chick-Watson)

$$
N(t) = N_0 e^{-kt}
$$

Python:
```python
def bacteria_inactivation(N0, k, t):
    return N0 * np.exp(-k * t)
```

## Salt Rejection

$$
\frac{C_p}{C_f} = 1 - R
$$

Python:
```python
def salt_rejection(Cf, R):
    return Cf * (1 - R)
```

# Add additional mechanisms as needed (e.g., intraparticle diffusion, competitive adsorption)
