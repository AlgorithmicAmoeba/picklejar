# first line: 96
@RunSequences.vectorize
@PickleJar.pickle(path='gsf/raw')
def resample_run_seq(N_particle, N_runs, gpu):
    """Performs a run sequence on the resample function with the given number
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
        gsf.weights = numpy.random.random(size=gsf.N_particles)
        gsf.weights /= numpy.sum(gsf.weights)
        t = time.time()
        gsf.resample()
        times.append(time.time() - t)

    return numpy.array(times)
