from pathlib import Path
import numpy as np

from ...constants import ARCSEC_TO_RAD

planetary_effect_cols = [
    "index",
    "(s,j)",
    "(c,j)",
    "l",
    "l'",
    "F",
    "D",
    "Om",
    "L_Me",
    "L_Ve",
    "L_E",
    "L_Ma",
    "L_J",
    "L_Sa",
    "L_U",
    "L_Ne",
    "p_A",
]


def import_table(filename: str):
    """Parse an IERS Conventions Table 5.2 file (CIP X, Y, or s series).

    Parameters
    ----------
    filename : str
        Name of the table file in this directory (e.g. `"tab5.2a.txt"`).

    Returns
    -------
    coeffs_microarcsec : np.ndarray (6,)
        Polynomial-part coefficients, constant term first [microarcsec].
    the_data : list of np.ndarray
        One array per power of `t` (`j = 0, 1, ...`). Each row holds the sine and
        cosine amplitudes [microarcsec] followed by the 14 fundamental-argument
        multipliers.

    Raises
    ------
    ImportError
        If the file is empty or is not a Table 5.2 file.
    """
    base = Path(__file__).resolve().parent
    filename = base / filename

    lines = []
    with open(filename, "r") as f:
        lines = f.readlines()

    num_lines = len(lines)
    if num_lines == 0 or not lines[0].startswith("Table 5.2"):
        raise ImportError(f"Imported empty file or non-CIP file: {filename}")

    # Index of reading
    i = 0

    # Find polynomial part
    while "Polynomial part (unit microarcsecond)" not in lines[i]:
        i += 1

    # Get coefficients
    i += 2
    coeffs_microarcsec = []
    sign = 1
    for symbol in lines[i].split(" "):
        if "t" in symbol or symbol == "":
            continue
        # Now just numbers and signs
        match symbol:
            case "+":
                sign = 1
            case "-":
                sign = -1
            case _:
                coeffs_microarcsec.append(float(symbol) * sign)

    the_data = []
    # Skip to where it says number of rows
    while len(lines) > 0:
        if "Number of terms" not in lines[0]:
            lines = lines[1:]
            continue

        # Parse num of rows
        num_terms = int(lines[0].split("Number of terms = ")[-1])
        lines = lines[2:]

        # planetary_effect_df = pd.DataFrame([l.split() for l in lines[:num_terms]], columns=planetary_effect_cols)

        # Starting at 1 gets rid of the simple index as well
        planetary_effects = np.array([l.split()[1:] for l in lines[:num_terms]])

        the_data.append(np.array(planetary_effects, dtype=float))
        lines = lines[num_terms:]
        # the_data.append(planetary_effect_df)

    # Returns in dataframes for sanity checking
    return np.asarray(coeffs_microarcsec), the_data


def approx_5th_deg_spline(t_tt, coeffs) -> float:
    """Evaluate the 5th-degree polynomial part of the IAU 2006 CIP series.

    Parameters
    ----------
    t_tt : float
        Julian centuries since J2000 in Terrestrial Time (TT).
    coeffs : array_like (6,)
        Polynomial coefficients, constant term first [microarcsec].

    Returns
    -------
    float
        Polynomial value [microarcsec].
    """
    return (
        coeffs[0]
        + coeffs[1] * t_tt
        + coeffs[2] * t_tt**2
        + coeffs[3] * t_tt**3
        + coeffs[4] * t_tt**4
        + coeffs[5] * t_tt**5
    )


FUNDAMENTAL_ARGUMENTS_ARCSEC = np.array(
    [
        # 0 - mean anomaly of the Moon
        [485868.249036, 1717915923.2178, 31.8792, 0.051635, -0.00024470],
        # 1 - mean anomaly of the Sun
        [1287104.79305, 129596581.0481, -0.5532, 0.000136, -0.00001149],
        # 2 - mean arguent of latitude of the Moon
        [335779.526232, 1739527262.8478, -12.7512, -0.001037, 0.00000417],
        # 3 - mean elongation of the Sun
        [1072260.70369, 1602961601.2090, -6.3706, 0.006593, -0.00003169],
        # 4 - right ascension of ascending node of mean lunar orbit
        [450160.398036, -6962890.5431, 7.4722, 0.007702, -0.00005939],
    ]
)

FUNDAMENTAL_ARGUMENTS_RAD = np.array(
    [
        # 5-12 - heliocentric latitudes of planets
        [4.402608842, 2608.7903141574, 0],  #  mercury
        [3.176146697, 1021.3285546211, 0],  #  venus
        [1.753470314, 628.3075849991, 0],  #  earth
        [6.203480913, 334.0612426700, 0],  #  mars
        [0.599546497, 52.9690962641, 0],  #  jupiter
        [0.874016757, 21.3299104960, 0],  #  saturn
        [5.481293872, 7.4781598567, 0],  #  uranus
        [5.311886287, 3.8133035638, 0],  #  neptune
        # 13 - general precession in longitude
        [0, 0.02438175, 0.00000538691],
    ]
)


def fundamental_arguments(t_tt: float):
    """Compute the 14 fundamental nutation arguments (Delaunay and planetary).

    Parameters
    ----------
    t_tt : float
        Julian centuries since J2000 in Terrestrial Time (TT).

    Returns
    -------
    args : np.ndarray (14,)
        Fundamental nutation arguments, wrapped to `[0, 2*pi)` [rad].

    References
    ----------
    Vallado 4e, Equations 3-58 and 3-59, p. 210-211.
    """
    t = t_tt
    t2 = t * t
    t3 = t * t2
    t4 = t * t3

    args_arcsec = (
        FUNDAMENTAL_ARGUMENTS_ARCSEC @ np.array([1, t, t2, t3, t4]) * ARCSEC_TO_RAD
    )
    args_rad = FUNDAMENTAL_ARGUMENTS_RAD @ np.array([1, t, t2])
    return np.hstack((args_arcsec, args_rad)) % (2 * np.pi)


# def trigonometric_argument(coeffs: np.ndarray, t_tt: float):
#     """Trigonometric argument `a_p_i` of the CIP series.

#     Parameters
#     ----------
#     coeffs : np.ndarray (N, 14)
#         Fundamental-argument multipliers from the nutation table.
#     t_tt : float
#         Julian centuries since J2000 in Terrestrial Time (TT).

#     Returns
#     -------
#     float
#         Trigonometric argument [rad].

#     References
#     ----------
#     Vallado 4e, Equation 3-63, p. 213.
#     """
#     args = fundamental_arguments(t_tt)
#     return coeffs * args


def get_summation(rows: np.ndarray, t_tt: float):
    """Evaluate one periodic summation term of the CIP X, Y, or s series.

    Parameters
    ----------
    rows : np.ndarray (N, 16)
        Table rows for a single power of `t`. Each row holds the sine and cosine
        amplitudes [microarcsec] followed by the 14 fundamental-argument multipliers.
    t_tt : float
        Julian centuries since J2000 in Terrestrial Time (TT).

    Returns
    -------
    float
        Sum of the periodic terms [microarcsec].

    Raises
    ------
    ImportError
        If a row does not have 16 entries.

    References
    ----------
    Vallado 4e, p. 214.
    """
    sum = 0
    for row in rows:
        if len(row) != 16:
            raise ImportError(
                "Imported row must have 2 sinusoidal parameters and 14 nutation parameters\n"
                f"Received: {row}"
            )

        a_p_i = np.dot(row[2:], fundamental_arguments(t_tt))  # element-wise

        sum += (row[0] * np.sin(a_p_i)) + (row[1] * np.cos(a_p_i))

    return sum


# def


# import_table("tab5.2a.txt")

# fundamental_arguments(0)
