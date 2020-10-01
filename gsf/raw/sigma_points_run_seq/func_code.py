# first line: 137
@RunSequences.vectorize
@PickleJar.pickle(path='gsf/raw')
def sigma_points_run_seq(N_particle, N_runs):
    """Performs a run sequence on the sigma point function with the given number
    of particle and number of runs

    Parameters
    ----------
    N_particle : int
        Number of particles

    N_runs : int
        Number of runs in the sequence

    Returns
    -------
    times : numpy.array
        The times of the run sequence
    """
    times = []

    _, _, _, gsf = sim_base.get_parts(
        N_particles=N_particle,
        gpu=True,
        pf=False
    )

    for _ in tqdm.tqdm(range(N_runs)):
        u, _ = sim_base.get_random_io()
        gsf.predict(u, 1.)

        t = time.time()
        # noinspection PyProtectedMember
        gsf._get_sigma_points()
        times.append(time.time() - t)

    return numpy.array(times)
