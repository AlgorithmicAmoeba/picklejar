# first line: 56
@RunSequences.vectorize
@PickleJar.pickle(path='gsf/raw')
def update_run_seq(N_particle, N_runs, gpu):
    """Performs a run sequence on the update function with the given number
    of particle and number of runs on the CPU or GPU

    Parameters
    ----------
    N_particle : int
        Number of particles

    N_runs : int
        Number of runs in the sequence

    gpu : bool
        If `True` then the GPU implementation is used.
        Otherwise, the CPU implementation is used

    Returns
    -------
    times : numpy.array
        The times of the run sequence
    """
    times = []

    _, _, _, gsf = sim_base.get_parts(
        N_particles=N_particle,
        gpu=gpu,
        pf=False
    )

    for j in range(N_runs):
        u, y = sim_base.get_random_io()
        t = time.time()
        gsf.update(u, y)
        times.append(time.time() - t)

    return numpy.array(times)
