"""Create the images used in the project documentation."""

from gaussian_field_excursions import simulate_field, plot_excursion, plot_field
from pathlib import Path
import matplotlib.pyplot as plt

def main():

    output_directory = (
        Path(__file__).resolve().parents[1] / "docs" / "images"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    field_bf = simulate_field(
        x_size=100,
        y_size=100,
        pixel_density=10,
        covariance="bf",
        seed=123
    )

    plot_excursion(
        field=field_bf,
        save=True,
        largest_comp=False,
        filename=output_directory / "bargmann_fock_excursion.png",
        show=False
    )
    
    plot_excursion(
        field=field_bf,
        save=True,
        filename=output_directory / "bargmann_fock_largest_component.png",
        show=False
    )

    field_rpw = simulate_field(
        x_size=100,
        y_size=100,
        pixel_density=10,
        covariance="RPW",
        seed=123
    )

    plot_excursion(
        field=field_rpw,
        threshold=0.5,
        save=True,
        largest_comp=False,
        filename=output_directory / "random_plane_wave_excursion.png",
        show=False
    )

    plot_excursion(
        field=field_rpw,
        largest_comp=False,
        boundary=True,
        thickness=5,
        save=True,
        filename=output_directory / "random_plane_wave_largest_component_boundary.png",
        show=False
    )

    field_matern = simulate_field(
        x_size=20,
        y_size=20,
        pixel_density=10,
        covariance="matern",
        seed=123
    )
    
    plot_field(
        field=field_matern,
        show=False,
        save=True,
        filename=output_directory / "matern_field.png"
    )

    #Add 
    # 1) black and white excursion set (different threshold)
    # 2) Example of different colour scheme (small number of pixels)

    plt.close("all")

if __name__ == "__main__":
    main()