# Gaussian Field Excursions

## Project purpose

## Installation

```bash
python -m venv .venv
pip install -r requirements.txt
```

## Minimal example

```python
from gaussian_field_excursions import plot_excursion, simulate_field

field = simulate_field(
    x_size=10,
    y_size=5,
    pixel_density=10,
    covariance="bf",
)

plot_excursion(field, boundary=True)
```

## Available covariance models

## Coordinate and length-scale conventions

## Licence

This project is available under the [MIT Licence](LICENSE).
