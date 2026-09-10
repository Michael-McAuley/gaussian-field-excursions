"""Simulation and visualisation of Gaussian-field excursion sets.

The individual stages are exposed as separate functions so that a simulated
field, its excursion set, its largest component, and the component boundary
can each be inspected or reused independently. ``plot_excursion`` remains as
a convenience wrapper around the complete workflow.
"""

from pathlib import Path

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from gstools import Gaussian, JBessel, Matern, Rational, SRF
from scipy import ndimage


DEFAULT_COLOURS = ("black", "white", "green", "orange")


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
    """Return the lower excursion set ``field > threshold`` as a boolean mask."""
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

    cmap = matplotlib.colors.ListedColormap(colours[:number_of_labels])
    boundaries = np.arange(number_of_labels + 1) - 0.5
    norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N)
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
