"""Create the images used in the project documentation."""

import gaussian_field_excursions as gfe
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def main():

    output_directory = (
        Path(__file__).resolve().parents[1] / "docs" / "images"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    field_bf = gfe.simulate_field(
        x_size=100,
        y_size=100,
        pixel_density=10,
        covariance="bf",
        seed=123
    )

    gfe.plot_excursion(
        field=field_bf,
        save=True,
        largest_comp=False,
        filename=output_directory / "bargmann_fock_excursion.png",
        show=False
    )
    
    gfe.plot_excursion(
        field=field_bf,
        save=True,
        filename=output_directory / "bargmann_fock_largest_component.png",
        show=False
    )

    field_rpw = gfe.simulate_field(
        x_size=100,
        y_size=100,
        pixel_density=10,
        covariance="RPW",
        seed=123
    )

    gfe.plot_excursion(
        field=field_rpw,
        threshold=0.5,
        save=True,
        largest_comp=False,
        filename=output_directory / "random_plane_wave_excursion.png",
        show=False
    )

    gfe.plot_excursion(
        field=field_rpw,
        largest_comp=False,
        boundary=True,
        thickness=5,
        save=True,
        filename=output_directory / "random_plane_wave_largest_component_boundary.png",
        show=False
    )

    field_matern = gfe.simulate_field(
        x_size=20,
        y_size=20,
        pixel_density=10,
        covariance="matern",
        seed=123
    )
    
    gfe.plot_excursion(
        field=field_matern,
        show=False,
        save=True,
        filename=output_directory / "matern_field.png"
    )

    thresholds = np.linspace(4, -4, 40)
    tensor_bf = gfe.excursion_set_tensor(field_bf, thresholds=thresholds)
    max_point_bf = np.unravel_index(np.argmax(field_bf), field_bf.shape)
    label_bf = gfe.label_component_point(tensor_bf, max_point_bf)
    gfe.animate_phase_transition(
        label_bf,
        filename=output_directory / "bargmann_fock_phase_transition.gif",
        threshold_values=thresholds,
        frame_duration_ms=200,
        colour_scheme={
            "mode": "selected",
            "colours": ("lightgrey", "crimson"),
            "background": "white",
        },
    )

    tensor_rpw = gfe.excursion_set_tensor(field_rpw, thresholds=thresholds)
    label_rpw = gfe.label_excursion_components(tensor_rpw)
    gfe.animate_phase_transition(
        label_rpw,
        filename=output_directory / "random_plane_wave_phase_transition.gif",
        threshold_values=thresholds,
        frame_duration_ms=200,
        colour_scheme={
            "mode": "components",
            "colours": "tab20",
            "background": "white",
        },
    )
    
    plt.close("all")

if __name__ == "__main__":
    main()