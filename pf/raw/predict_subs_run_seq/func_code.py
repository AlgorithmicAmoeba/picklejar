# first line: 134
@RunSequences.vectorize
@PickleJar.pickle(path='pf/raw')
def predict_subs_run_seq(N_particle, N_runs):
    """Performs a run sequence on the prediction function's subroutines
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

    dt = 1.
    for _ in tqdm.tqdm(range(N_runs)):
        u, _ = sim_base.get_random_io()
        pf.predict(u, 1.)

        times = []
        t = time.time()
        u = cupy.asarray(u)
        times.append(time.time() - t)

        t = time.time()
        pf.particles += pf.f_vectorize(pf.particles, u, dt)
        times.append(time.time() - t)

        t = time.time()
        pf.particles += pf.state_pdf.draw(pf.N_particles)
        times.append(time.time() - t)

        timess.append(times)

    return numpy.array(timess)
