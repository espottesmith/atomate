from pymatgen.io.qchem.outputs import QCOutput


class ImproperTermination(Exception):
    """Raised when the Gaussian output did not properly terminate"""
    pass


class NoFrequencies(Exception):
    """Raised when the Gaussian output parser did not find frequencies"""
    pass


class NoImaginaryFrequencies(Exception):
    """Raised when there are no imaginary frequencies in the Freq output"""
    pass


class MultipleImaginaryFrequencies(Exception):
    """Raised when there are are multiple imaginary frequencies in the Freq output"""
    pass


def get_imaginary_freq_mode(output):
    """
    Returns the imaginary frequency mode from a Q-Chem frequency calculation.

    Args:
        output: QCOutput object for a frequency calculation

    Returns:
        mode: np.ndarray representing a translation vector to follow the
            imaginary mode
    """

    if not isinstance(output, QCOutput):
        raise ValueError("get_imaginary_freq_mode requires a QCOutput file.")

    if not output.data["completion"]:
        raise ImproperTermination("Calculation is not complete.")
    if "frequencies" in output.data:
        if output.data["frequencies"] is None:
            raise NoFrequencies("No frequencies parsed.")
        if output.data["frequencies"][0] >= 0:
            raise NoImaginaryFrequencies('No imaginary frequencies.')
        if output.data["frequencies"][1] < 0:
            raise MultipleImaginaryFrequencies('Multiple imaginary frequencies')
    else:
        raise NoFrequencies("No frequencies parsed.")

    mode = output.data["frequency_mode_vectors"][0]
    return mode


def perturb_single_mode(coords, mode, magnitude=0.6):
    """
    Perturb a set of molecular coordinates along a frequency mode.

    Args:
        coords: np.ndarray representing molecular coordinates
        mode: np.ndarray representing a translation vector for a particular
            frequency mode
        magnitude: float scaling factor for the translation vector

    Returns:
        [p1_coords, p2_coords]: perturbed molecule coordinates in both directions
    """

    p1_coords = coords + magnitude * mode
    p2_coords = coords - magnitude * mode

    return [p1_coords, p2_coords]