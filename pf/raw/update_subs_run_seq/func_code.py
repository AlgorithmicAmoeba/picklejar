# first line: 185
@RunSequences.vectorize
@PickleJar.pickle(path='pf/raw')
def update_subs_run_seq(N_particle, N_runs):
    """Performs a run sequence on the update function's subroutines
     with the given number of particles and number of runs

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
    timess = []

    _, _, _, pf = sim_base.get_parts(
        N_particles=N_particle,
        gpu=True,
        pf=True
    )

    for _ in tqdm.tqdm(range(N_runs)):
        u, z = sim_base.get_random_io()
        pf.predict(u, 1.)

        times = []
        t = time.time()
        u = cupy.asarray(u)
        z = cupy.asarray(z, dtype=cupy.float32)
        times.append(time.time() - t)

        t = time.time()
        ys = cupy.asarray(pf.g_vectorize(pf.particles, u, pf._y_dummy))
        times.append(time.time() - t)

        t = time.time()
        es = z - ys
        ws = cupy.asarray(pf.measurement_pdf.pdf(es))
        pf.weights *= ws
        times.append(time.time() - t)

        timess.append(times)

    return numpy.array(timess)
