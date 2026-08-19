from gaussian_field_excursions import plot_excursion, simulate_field


if __name__ == "__main__":
    field = simulate_field(
        x_size=20,
        y_size=10,
        pixel_density=10,
        covariance="bf",
    )

    plot_excursion(
        field,
        boundary=True,
    )
