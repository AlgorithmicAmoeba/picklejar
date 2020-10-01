# first line: 129
@RunSequences.vectorize
@PickleJar.pickle(path='gsf/raw')
@PowerMeasurement.measure
def nothing_power_seq(N_particle, t_run):
    """Performs a power sequence on the no-op function with the given number
    of particle and number of runs on the CPU or GPU.
    Used to check default power usage

    Parameters
    ----------
    N_particle : int
        Number of particles

    t_run : float
        Minimum run time of the function. Repeats if the time is too short

    Returns
    -------
    runs : int
        The number of times the function was run
    """
    _ = N_particle
    t = time.time()
    runs = 0
    while time.time() - t < t_run:
        runs += 1
        time.sleep(246)

    return runs
