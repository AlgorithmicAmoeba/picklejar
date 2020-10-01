# first line: 48
@RunSequences.vectorize
@PickleJar.pickle(path='pf/raw')
@PowerMeasurement.measure
def update_power_seq(N_particle, t_run, gpu):
    """Performs a power sequence on the update function with the given number
    of particle and number of runs on the CPU or GPU

    Parameters
    ----------
    N_particle : int
        Number of particles

    t_run : float
        Minimum run time of the function. Repeats if the time is too short

    gpu : bool
        If `True` then the GPU implementation is used.
        Otherwise, the CPU implementation is used

    Returns
    -------
    runs : int
        The number of times the function was run
    """
    _, _, _, p = sim_base.get_parts(
        N_particles=N_particle,
        gpu=gpu
    )

    t = time.time()
    runs = 0
    while time.time() - t < t_run:
        runs += 1
        u, y = sim_base.get_random_io()
        p.update(u, y)

    return runs
