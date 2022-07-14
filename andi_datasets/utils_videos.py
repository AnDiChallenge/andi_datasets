# AUTOGENERATED! DO NOT EDIT! File to edit: source_nbs/utils_videos.ipynb (unless otherwise specified).

__all__ = ['func_poisson_noise', 'CIRCLE_RADIUS', 'circle', 'circle', 'get_video_andi', 'play_video']

# Cell
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import numpy as np

import deeptrack as dt

# Cell
def func_poisson_noise():
    def inner(image):
        image[image<0] = 0
        rescale = 1
        noisy_image = np.random.poisson(image * rescale) / rescale
        return noisy_image
    return inner

# Cell
# For generating masks
CIRCLE_RADIUS = 1
X, Y = np.mgrid[:2*CIRCLE_RADIUS+1, :2*CIRCLE_RADIUS+1]
circle = (X - CIRCLE_RADIUS)**2 + (Y- CIRCLE_RADIUS)**2 < CIRCLE_RADIUS**2
circle = np.expand_dims(circle, axis=-1)

# Cell
def get_video_andi(
    trajectory_data, particle_props={}, optics_props={}, background_props={}
):
    """Generates a video from a trajectory data.

    Function needs to called with update().resolve() to create the video.

    Parameters
    ----------
    trajectory_data : np.ndarray
        Generated through models_phenom. Array of shape (T, N, 2) containing the trajectories.
    particle_props : dict
        Dictionary of properties for the particles.
    optics_props : dict
        Dictionary of properties for the optics.
    background_props : dict
        Dictionary of properties for the background.
    """

    _particle_dict = {
        "particle_intensity": [100, 20],                                # Mean and standard deviation of the particle intensity
        "intensity": lambda particle_intensity: particle_intensity[0]
        + np.random.randn() * particle_intensity[1],
        "intensity_variation": 5,                                       # Intensity variation of particle (in standard deviation)
        "z": 0,                                                         # Particles are always at focus - this shouldn't be changed
        "refractive_index": 1.45,                                       # Refractive index of the particle
        "position_unit": "pixel",
    }

    _optics_dict = {
        "NA": 1.46,                 # Numerical aperture
        "wavelength": 500e-9,       # Wavelength
        "resolution": 100e-9,       # Camera resolution or effective resolution
        "magnification": 1,
        "upscale": 4,
        "refractive_index_medium": 1.33,
        "output_region": [0, 0, 128, 128],
    }

    # Background offset
    _background_dict = {
        "background_mean": 10,      # Mean background intensity
        "background_std": 2,        # Standard deviation of background intensity within a video
    }

    # Update the dictionaries with the user-defined values
    _particle_dict.update(particle_props)
    _optics_dict.update(optics_props)
    _background_dict.update(background_props)

    # Reshape the trajectory
    trajectory_data = np.moveaxis(trajectory_data, 0, 1)

    # Generate point particles
    particle = dt.PointParticle(
        trajectories=trajectory_data,
        replicate_index=lambda _ID: _ID,
        trajectory=lambda replicate_index, trajectories: dt.units.pixel
        * trajectories[replicate_index[-1]],
        number_of_particles=trajectory_data.shape[0],
        traj_length=trajectory_data.shape[1],
        position=lambda trajectory: trajectory[0],
        **_particle_dict,
    )

    # Intensity variation of particles - controlled by "intensity_variation"
    def intensity_noise(previous_values, previous_value):
        return (previous_values or [previous_value])[0] + _particle_dict[
            "intensity_variation"
        ] * np.random.randn()

    # Make it sequential
    sequential_particle = dt.Sequential(
        particle,
        position=lambda trajectory, sequence_step: trajectory[sequence_step],
        intensity=intensity_noise,
    )

    # Adding background offset
    background = dt.Add(
        value=_background_dict["background_mean"]
        + np.random.randn() * _background_dict["background_std"]
    )

    def background_variation(previous_values, previous_value):
        return (previous_values or [previous_value])[
            0
        ] + np.random.randn() * _background_dict["background_std"]

    ## This will change the background offset within a sequence with a given standard deviation
    sequential_background = dt.Sequential(background, value=background_variation)

    # Define optical setup
    optics = dt.Fluorescence(**_optics_dict)

    # Scale factor for image plane peak intensity
    scale_image = dt.Multiply(20)

    # Poisson noise
    poisson_noise = dt.Lambda(func_poisson_noise)

    # Sample
    sample = (
        optics(sequential_particle ^ sequential_particle.number_of_particles)
        >> scale_image
        >> sequential_background
        >> poisson_noise
    )

    # Masks
    get_masks = dt.SampleToMasks(
        lambda: lambda image: circle,
        output_region=optics.output_region,
        merge_method="or",
    )

    masks = sample >> get_masks

    # Sequential sample
    sequential_sample = dt.Sequence(
        (sample & masks),
        trajectory=particle.trajectories,
        sequence_length=particle.traj_length,
    )

    return sequential_sample

# Cell
def play_video(video, figsize=(5, 5), fps=10):
    """Visualizes the stack of images.

    Parameters
    ----------
    video : ndarray
        Stack of images.
    figsize : tuple, optional
        Size of the figure.
    fps : int, optional
        Frames per second.
    """

    fig = plt.figure(figsize=figsize)
    images = []
    plt.axis("off")

    for image in video:
        images.append([plt.imshow(image[:, :, 0], cmap="gray")])

    anim = animation.ArtistAnimation(
        fig, images, interval=1e3 / fps, blit=True, repeat_delay=0
    )

    return HTML(anim.to_jshtml())