"""Simulation and visualisation of Gaussian-field excursion sets.

The individual stages are exposed as separate functions so that a simulated
field, its excursion set, its largest component, and the component boundary
can each be inspected or reused independently. ``plot_excursion`` remains as
a convenience wrapper around the complete workflow.
"""

from pathlib import Path
from collections.abc import Mapping
from matplotlib import animation, colors, colormaps
import matplotlib.pyplot as plt
import numpy as np
from gstools import Gaussian, JBessel, Matern, Rational, SRF
from scipy import ndimage



DEFAULT_COLOURS = ("black", "white", "green", "orange")
DEFAULT_ANIMATION_COLOUR_SCHEME = {
    "mode": "components",
    "colours": "tab20",
    "background": "white",
}


def _covariance_model(covariance, beta=1, nu=1):
    """Build the requested covariance model and return it with a display name."""
    covariance_key = str(covariance).strip().lower()

    if covariance_key == "matern":
        return (
            Matern(dim=2, var=1, len_scale=1, nu=nu),
            "Matern",
        )
    if covariance_key == "bf":
        return Gaussian(dim=2, var=1, len_scale=1), "Bargmann-Fock"
    if covariance_key == "rq":
        return (
            Rational(dim=2, var=1, len_scale=1, alpha=beta),
            "Rational-Quadratic",
        )
    if covariance_key == "rpw":
        return JBessel(dim=2, var=1, len_scale=1, nu=0), "RPW"

    raise ValueError("covariance must be one of 'RPW', 'matern', 'bf', or 'rq'")


def _simulation_grid(x_size, y_size, pixel_density):
    """Return uniformly spaced coordinate vectors for the simulation domain."""
    for value, name in (
        (x_size, "x_size"),
        (y_size, "y_size"),
        (pixel_density, "pixel_density"),
    ):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")

    number_of_x_pixels = int(np.ceil(x_size * pixel_density))
    number_of_y_pixels = int(np.ceil(y_size * pixel_density))
    x_coordinates = np.arange(number_of_x_pixels) / pixel_density
    y_coordinates = np.arange(number_of_y_pixels) / pixel_density
    return x_coordinates, y_coordinates


def simulate_field(
    x_size,
    y_size,
    pixel_density,
    covariance="RPW",
    beta=6,
    nu=1,
    seed=42,
):
    """Simulate a Gaussian field on a rectangular, uniformly spaced grid.

    ``x_size`` and ``y_size`` are the physical dimensions of the half-open
    simulation domain. ``pixel_density`` is the number of samples per unit in
    both directions, so the grid spacing is ``1 / pixel_density``. The returned
    array's axes are ordered as ``(x, y)``. Length is measured in covariance-
    length units, with every covariance model's length scale fixed at one.
    """
    x_coordinates, y_coordinates = _simulation_grid(
        x_size, y_size, pixel_density
    )
    model, _ = _covariance_model(covariance, beta, nu)
    return SRF(model, seed=seed).structured([x_coordinates, y_coordinates])


def find_excursion_set(field, threshold=0):
    """Return the upper excursion set ``field > threshold`` as a boolean mask."""
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError("field must be a two-dimensional array")
    return field > threshold


def find_largest_component(excursion_set):
    """Return the largest four-connected component as a boolean mask.

    If the excursion set is empty, the returned mask is empty as well. When
    several components have the same size, the first one in array order is
    selected.
    """
    excursion_mask = _boolean_mask(excursion_set, "excursion_set")
    four_neighbours = ndimage.generate_binary_structure(rank=2, connectivity=1)
    labels, number_of_components = ndimage.label(
        excursion_mask, structure=four_neighbours
    )

    if number_of_components == 0:
        return np.zeros_like(excursion_mask, dtype=bool)

    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0  # Label zero is the non-excursion background.
    largest_label = component_sizes.argmax()
    return labels == largest_label


def find_boundary(largest_component):
    """Return the one-pixel inner boundary of a component as a boolean mask.

    Four-neighbour connectivity is used, consistently with
    ``find_largest_component``. Component pixels at the image edge are
    correctly included in the boundary.
    """
    component_mask = _boolean_mask(largest_component, "largest_component")
    four_neighbours = ndimage.generate_binary_structure(rank=2, connectivity=1)
    interior = ndimage.binary_erosion(
        component_mask,
        structure=four_neighbours,
        border_value=0,
    )
    return component_mask & ~interior


def plot_field(
    field,
    ax=None,
    show=True,
    save=False,
    filename="",
):
    """Plot a two-dimensional field using a continuous colour scale."""
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError("field must be a two-dimensional array")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.imshow(field.T, interpolation="nearest")
    return _finish_plot(fig, ax, show, save, filename, "field.png")


def plot_mask(
    mask,
    colours=DEFAULT_COLOURS,
    ax=None,
    show=True,
    save=False,
    filename="",
):
    """Plot a two-dimensional Boolean or non-negative integer-labelled mask.

    Each integer label is mapped to the colour at the corresponding position
    in ``colours``. Boolean masks are treated as masks with labels zero and one.
    Pipeline arrays are transposed for display so their x-axis is horizontal
    and their y-axis is vertical.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    if mask.size == 0:
        raise ValueError("mask must not be empty")
    if not (
        np.issubdtype(mask.dtype, np.bool_)
        or np.issubdtype(mask.dtype, np.integer)
    ):
        raise TypeError("mask must contain Boolean or integer labels")
    if np.any(mask < 0):
        raise ValueError("mask labels must be non-negative")

    labelled_mask = mask.astype(np.int64, copy=False)
    number_of_labels = int(labelled_mask.max()) + 1
    if len(colours) < number_of_labels:
        raise ValueError(
            f"colours must contain at least {number_of_labels} colours"
        )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    cmap = colors.ListedColormap(colours[:number_of_labels])
    boundaries = np.arange(number_of_labels + 1) - 0.5
    norm = colors.BoundaryNorm(boundaries, cmap.N)
    ax.imshow(
        labelled_mask.T,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    return _finish_plot(fig, ax, show, save, filename, "mask.png")


def _boolean_mask(array, name):
    """Validate a two-dimensional input and convert it to a boolean mask."""
    mask = np.asarray(array, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    return mask


def _thicken_boundary(boundary, radius):
    """Dilate a boundary with a disk of the given pixel radius."""
    if radius <= 0:
        return boundary
    coordinates = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(coordinates, coordinates, indexing="ij")
    disk = xx**2 + yy**2 <= radius**2
    return ndimage.binary_dilation(boundary, structure=disk)


def _finish_plot(fig, ax, show, save, filename, default_filename):
    """Apply common figure formatting, saving, and display behaviour."""
    ax.set_xticks([])
    ax.set_yticks([])

    if save:
        fig.savefig(Path(filename or default_filename), bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_excursion(
    field,
    threshold=0,
    save=False,
    largest_comp=True,
    boundary=False,
    thickness=3,
    colours=DEFAULT_COLOURS,
    filename="",
    show=True,
):
    """Plot the requested excursion-set layers for a simulated field.

    The excursion set is always shown; ``largest_comp`` highlights its largest
    component and ``boundary`` highlights that component's boundary. The field
    must already have been generated, for example by ``simulate_field``. The
    intermediate arrays are returned in a dictionary so callers can reuse them.
    """
    if thickness < 0:
        raise ValueError("thickness must be non-negative")

    field = np.asarray(field)
    excursion_set = find_excursion_set(field, threshold=threshold)

    largest_component = None
    boundary_set = None
    display = excursion_set.astype(np.uint8)

    if largest_comp or boundary:
        largest_component = find_largest_component(excursion_set)
        if largest_comp:
            display[largest_component] = 2
        if boundary:
            boundary_set = find_boundary(largest_component)
            radius = int(np.ceil(thickness))
            displayed_boundary = _thicken_boundary(boundary_set, radius)
            display[displayed_boundary] = 3

    number_of_colours = 4 if boundary else (3 if largest_comp else 2)
    if len(colours) < number_of_colours:
        raise ValueError(
            f"colours must contain at least {number_of_colours} colours"
        )

    if save and not filename:
        x_pixels, y_pixels = field.shape
        filename = (
            f"excursion_{x_pixels}x{y_pixels}px_"
            f"threshold={threshold}.png"
        )

    fig, ax = plot_mask(
        display,
        colours=colours[:number_of_colours],
        show=show,
        save=save,
        filename=filename,
    )

    return {
        "field": field,
        "excursion_set": excursion_set,
        "largest_component": largest_component,
        "boundary": boundary_set,
        "figure": fig,
        "axes": ax,
    }


def excursion_set_tensor(field, thresholds):
    """Returns a tensor of excursion sets for the given field at the specified thresholds."""
    tensor = np.zeros((len(thresholds), *field.shape), dtype=bool)
    for i, threshold in enumerate(thresholds):
        tensor[i] = field > threshold
    return tensor

def label_excursion_components(tensor):
    """Labels connected components in the excursion set tensor."""
    labelled_tensor = np.zeros_like(tensor, dtype=int)

    labelled_tensor[0], num_features = ndimage.label(tensor[0])
    for i in range(1, tensor.shape[0]):
        current_components, current_count = ndimage.label(tensor[i])
        for j in range(1, current_count + 1):
            # Find the overlap with the previous level's labels
            overlap = labelled_tensor[i-1][(current_components == j) & (labelled_tensor[i-1] > 0)]
            if overlap.size > 0:
                # Assign the most common label from the previous level
                most_common_label = np.bincount(overlap).argmax()
                labelled_tensor[i][current_components == j] = most_common_label
            else:
                # Assign a new label
                num_features += 1
                labelled_tensor[i][current_components == j] = num_features
    return labelled_tensor

def label_largest_component(tensor):
    """Label the largest component at each level of an excursion-set tensor.

    Background pixels receive label zero, pixels in all non-largest excursion
    components receive label one, and pixels in the largest component receive
    label two.
    """
    tensor = np.asarray(tensor, dtype=bool)
    if tensor.ndim != 3:
        raise ValueError("tensor must have shape (frames, rows, columns)")

    labelled_tensor = np.zeros(tensor.shape, dtype=np.uint8)
    labelled_tensor[tensor] = 1

    for i in range(tensor.shape[0]):
        largest_component = find_largest_component(tensor[i])
        labelled_tensor[i][largest_component] = 2

    return labelled_tensor

def label_component_point(tensor, point):
    """Label the component containing a specific point in an excursion-set tensor.

    Background pixels receive label zero, pixels in all other components receive
    label one, and pixels in the component containing the specified point receive
    label two.
    """
    tensor = np.asarray(tensor, dtype=bool)
    if tensor.ndim != 3:
        raise ValueError("tensor must have shape (frames, rows, columns)")

    labelled_tensor = np.zeros(tensor.shape, dtype=np.uint8)
    labelled_tensor[tensor] = 1

    for i in range(tensor.shape[0]):
        if tensor[i][point]:
            # Label the component containing the point
            component = ndimage.label(tensor[i])[0]
            point_label = component[point]
            labelled_tensor[i][component == point_label] = 2

    return labelled_tensor

def _validate_animation_inputs(
    labelled_tensor, filename, frame_duration_ms, threshold_values
):
    """Validate and normalise the common animation inputs."""
    labelled_tensor = np.asarray(labelled_tensor)
    if labelled_tensor.ndim != 3:
        raise ValueError(
            "labelled_tensor must have shape (frames, rows, columns)"
        )
    if labelled_tensor.shape[0] == 0:
        raise ValueError("labelled_tensor must contain at least one frame")
    if not np.issubdtype(labelled_tensor.dtype, np.integer):
        raise TypeError("labelled_tensor must contain integer labels")
    if np.any(labelled_tensor < 0):
        raise ValueError("labels must be non-negative integers")
    if frame_duration_ms <= 0:
        raise ValueError("frame_duration_ms must be positive")
    if (
        threshold_values is not None
        and len(threshold_values) != labelled_tensor.shape[0]
    ):
        raise ValueError("threshold_values must contain one value per frame")

    output_path = Path(filename)
    if output_path.suffix.lower() != ".gif":
        raise ValueError("filename must have a .gif extension")

    return labelled_tensor, output_path


def _label_colour_mapping(labelled_tensor, colour_scheme=None):
    """Return the colour map and norm described by ``colour_scheme``."""
    if colour_scheme is None:
        colour_scheme = DEFAULT_ANIMATION_COLOUR_SCHEME
    if not isinstance(colour_scheme, Mapping):
        raise TypeError("colour_scheme must be a mapping")

    unknown_keys = set(colour_scheme) - {"mode", "colours", "background"}
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown colour_scheme key(s): {unknown}")

    mode = str(colour_scheme.get("mode", "components")).strip().lower()
    if mode not in {"components", "selected", "uniform"}:
        raise ValueError(
            "colour_scheme mode must be 'components', 'selected', or 'uniform'"
        )

    background = colors.to_rgba(
        colour_scheme.get("background", "white")
    )
    maximum_label = int(labelled_tensor.max())

    if mode == "components":
        selected_colour_map = colormaps.get_cmap(
            colour_scheme.get("colours", "tab20")
        )
        positive_colours = (
            selected_colour_map(np.linspace(0, 1, maximum_label))
            if maximum_label > 0
            else np.empty((0, 4))
        )
    elif mode == "selected":
        if not np.isin(labelled_tensor, (0, 1, 2)).all():
            raise ValueError(
                "In selected mode, labelled_tensor may only contain "
                "labels 0, 1, and 2"
            )
        selected_colours = colour_scheme.get(
            "colours", ("lightgrey", "crimson")
        )
        if (
            isinstance(selected_colours, (str, bytes))
            or len(selected_colours) != 2
        ):
            raise ValueError(
                "Selected mode requires exactly two colours: "
                "(other, selected)"
            )
        positive_colours = np.array(
            [colors.to_rgba(colour) for colour in selected_colours]
        )
    else:
        uniform_colour = colors.to_rgba(
            colour_scheme.get("colours", "tab:blue")
        )
        positive_colours = np.tile(uniform_colour, (maximum_label, 1))

    colour_values = np.vstack((background, positive_colours))
    discrete_colour_map = colors.ListedColormap(colour_values)
    boundaries = np.arange(discrete_colour_map.N + 1) - 0.5
    colour_norm = colors.BoundaryNorm(boundaries, discrete_colour_map.N)
    return discrete_colour_map, colour_norm


def animate_phase_transition(
    labelled_tensor,
    filename="labelled_excursion_sets.gif",
    frame_duration_ms=500,
    threshold_values=None,
    colour_scheme=None,
):
    """Create a GIF showing changes in a labelled excursion-set tensor.

    The first axis is interpreted as the frame axis. Label zero is the
    background, and each positive label keeps the same colour in every frame.

    Parameters
    ----------
    labelled_tensor : numpy.ndarray
        A three-dimensional integer array with shape (frames, rows, columns).
    filename : str or pathlib.Path, optional
        Output GIF path.
    frame_duration_ms : int or float, optional
        Time for which each frame is displayed, in milliseconds.
    threshold_values : sequence, optional
        Thresholds to display in the frame titles. Its length
        must equal the number of frames.
    colour_scheme : mapping, optional
        Colour configuration with ``mode``, ``colours``, and ``background``
        entries. Mode ``"components"`` assigns a colour-map colour to each
        positive label; ``"selected"`` maps labels one and two to the
        ``(other, selected)`` colours; and ``"uniform"`` gives every positive
        label the same colour. By default, components use ``"tab20"`` on a
        white background.

    Returns
    -------
    pathlib.Path
        The path of the saved GIF.
    """

    labelled_tensor, output_path = _validate_animation_inputs(
        labelled_tensor, filename, frame_duration_ms, threshold_values
    )
    number_of_frames = labelled_tensor.shape[0]

    discrete_colour_map, colour_norm = _label_colour_mapping(
        labelled_tensor, colour_scheme=colour_scheme
    )

    has_title = threshold_values is not None
    rows, columns = labelled_tensor.shape[1:]
    image_width = 6
    image_height = image_width * rows / columns
    title_height = 0.4 if has_title else 0

    figure, axis = plt.subplots(
        figsize=(image_width, image_height + title_height)
    )
    figure.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=image_height / (image_height + title_height),
    )
    image = axis.imshow(
        labelled_tensor[0],
        cmap=discrete_colour_map,
        norm=colour_norm,
        interpolation="nearest",
    )
    axis.set_axis_off()
    title = axis.set_title(
        f"Threshold: {threshold_values[0]:.2f}"
        if has_title else "",
        pad=3,
    )

    def update(frame):
        image.set_data(labelled_tensor[frame])
        if threshold_values is not None:
            title.set_text(f"Threshold: {threshold_values[frame]:.2f}")
        return image, title

    gif_animation = animation.FuncAnimation(
        figure,
        update,
        frames=number_of_frames,
        interval=frame_duration_ms,
        blit=True,
    )
    writer = animation.PillowWriter(fps=1000 / frame_duration_ms)
    gif_animation.save(output_path, writer=writer)
    plt.close(figure)

    return output_path
