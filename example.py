from gaussian_field_excursions import simulate_field, plot_excursion, excursion_set_tensor, label_excursion_components, animate_phase_transition


if __name__ == "__main__":
    field = simulate_field(
        x_size=20,
        y_size=10,
        pixel_density=10,
        covariance="bf",
        seed=123
    )

    plot_excursion(
        field,
        boundary=True,
    )

    thresholds = [1, 0.5, 0, -0.5, -1]
    tensor = excursion_set_tensor(field, thresholds=thresholds)
    component_labels = label_excursion_components(tensor)
    
    animate_phase_transition(
        component_labels,
        filename="example_phase_transition.gif"
    )

