import numpy as np
import gnss_lib_py as glp


def get_closest_rinex_data(gps_millis, rinex_nav, constellations=None):
    """Find the appropriate closest rinex entry for the given gps_millis.

    For GPS, Galileo, Beidou, QZSS, and IRNSS, ephemeris parameters for
    the closest time in the past are used. For GLONASS and SBAS, the
    ephemeris parameters that are closest in time are used. In some cases,
    as per the RINEX v3 specification [1]_, this includes using parameters
    from the near future.

    Parameters
    ----------
    gps_millis : float
        Time at which SV states will be estimated and for which closest
        ephemeris parameters are needed.
    rinex_nav : gnss_lib_py.parsers.rinex_nav.RinexNav
        RinexNav object containing ephemeris parameters for all the
        constellations that are required.

    Returns
    -------
    trimmed_rinex : gnss_lib_py.parsers.navdata.NavData
        `NavData` instance containing `RinexNav` rows with the correspondingly
        trimmed epheemeris parameters for SV state estimation.

    References
    ----------
    ..  [1] Rinex Specification Document
        http://acc.igs.org/misc/rinex304.pdf. Retrieved on 23rd August,
        2023.
    """
    if constellations is not None:
        if isinstance(constellations, str):
            constellations = [constellations]
        rinex_nav = rinex_nav.where("gnss_id", constellations, "eq")
    rinex_gps_like, rinex_glonass_like = _split_rinex_nav(rinex_nav)
    if len(rinex_glonass_like) != 0:
        rinex_glonass_like["time_diff"] = np.abs(rinex_glonass_like["gps_millis"] - gps_millis)
        rinex_glonass_like_df = rinex_glonass_like.pandas_df()
        closest_glonass_like_df = rinex_glonass_like_df.loc[
            rinex_glonass_like_df.groupby("gnss_sv_id")["time_diff"].idxmin()
        ]
        closest_glonass_like = glp.NavData(pandas_df=closest_glonass_like_df.reset_index(drop=True))
        closest_glonass_like.remove(rows="time_diff", inplace=True)
    else:
        closest_glonass_like = glp.NavData()
    # Split the rinex data into other constellations and find the closest
    # time, in the past, to the query time
    if len(rinex_gps_like) != 0:
        time_cropped_gps_like = rinex_gps_like.where("gps_millis", gps_millis, "lesser")
        # Convert back to pd.DataFrame to utilize the strong sort_values
        # functionality
        time_cropped_gps_like = (
            time_cropped_gps_like.pandas_df().sort_values("gps_millis").groupby("gnss_sv_id").last()
        )
        time_cropped_gps_like = glp.NavData(pandas_df=time_cropped_gps_like.reset_index())
    else:
        time_cropped_gps_like = glp.NavData()
    trimmed_rinex = glp.concat(time_cropped_gps_like, closest_glonass_like)

    return trimmed_rinex


def _split_rinex_nav(rinex_nav):
    """Split the RinexNav instance into GPS like and GLONASS like parts.

    The GPS like part contains parameters for constellations that use
    orbital parameters for SV state estimation. The GLONASS like part
    contains parameters for constellations that use numerical integration
    for SV state estimation.

    Parameters
    ----------
    rinex_nav : gnss_lib_py.parsers.rinex_nav.RinexNav
        RinexNav instance containing ephemeris parameters, with one set
        of parameters per single satellite.

    Returns
    -------
    orbit_param_rinex : gnss_lib_py.parsers.rinex_nav.RinexNav
        RinexNav like instance of NavData containing orbital paramters
        for SV estimation of GPS like satellites.
    num_int_rinex : gnss_lib_py.parsers.rinex_nav.RinexNav
        RinexNav like instance of NavData containing numerical integration
        paramters for SV estimation of GLONASS like satellites.
    """
    orbit_param_rinex = rinex_nav.where("gnss_id", ["glonass", "sbas"], "neq")
    num_int_rinex = rinex_nav.where("gnss_id", ["glonass", "sbas"], "eq")
    return orbit_param_rinex, num_int_rinex


def check_eccentric_anomaly_calculation(func):
    """Test the Newton-Raphson calculation of the eccentric anomaly."""

    # Generate eccentricities between 0 and 0.9.
    ecc_vec = np.linspace(0, 0.9)

    # Generate eccentric anomalies between 0 and 2 pi.
    e_anom_vec = np.linspace(0, 2 * np.pi, 103)

    # Per eccentricity
    for ecc in ecc_vec:

        # Per true eccentric anomaly
        for ecc_anomaly in e_anom_vec:
            # Calculate the corresponding mean anomalies
            mean_anomaly = ecc_anomaly - ecc * np.sin(ecc_anomaly)

            # Check that we can recover the eccentric anomalies
            ecc_anomaly_recovered = func(mean_anomaly, ecc, 1e-5, 10)

            # Check that the recovered E matches the original E
            np.testing.assert_allclose(
                ecc_anomaly,
                ecc_anomaly_recovered,
                err_msg=f"\nThe true eccentric anomaly ({ecc_anomaly}) is not "
                + "sufficiently close to the recovered eccentric "
                + f"anomaly ({ecc_anomaly_recovered}) for eccentricity {ecc}\n",
            )
            # print(counter, ecc, e_anomaly)

    print("All tests passed.")


def check_eccentric_anomaly_calculation(func):
    """Test the Newton-Raphson calculation of the eccentric anomaly."""

    # Generate eccentricities between 0 and 0.9.
    ecc_vec = np.linspace(0, 0.9)

    # Generate eccentric anomalies between 0 and 2 pi.
    e_anom_vec = np.linspace(0, 2 * np.pi, 103)

    # Per eccentricity
    for ecc in ecc_vec:

        # Per true eccentric anomaly
        for ecc_anomaly in e_anom_vec:
            # Calculate the corresponding mean anomalies
            mean_anomaly = ecc_anomaly - ecc * np.sin(ecc_anomaly)

            # Check that we can recover the eccentric anomalies
            ecc_anomaly_recovered = func(mean_anomaly, ecc, 1e-5, 10)

            # Check that the recovered E matches the original E
            np.testing.assert_allclose(
                ecc_anomaly,
                ecc_anomaly_recovered,
                err_msg=f"\nThe true eccentric anomaly ({ecc_anomaly}) is not "
                + "sufficiently close to the recovered eccentric "
                + f"anomaly ({ecc_anomaly_recovered}) for eccentricity {ecc}\n",
            )
            # print(counter, ecc, e_anomaly)

    print("All tests passed.")
