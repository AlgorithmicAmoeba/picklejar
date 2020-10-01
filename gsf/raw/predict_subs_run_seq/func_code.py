# first line: 177
@RunSequences.vectorize
@PickleJar.pickle(path='gsf/raw')
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

    _, _, _, gsf = sim_base.get_parts(
        N_particles=N_particle,
        gpu=True,
        pf=False
    )

    dt = 1.
    for _ in tqdm.tqdm(range(N_runs)):
        u, _ = sim_base.get_random_io()
        gsf.predict(u, 1.)

        times = []
        t = time.time()
        sigmas = gsf._get_sigma_points()
        times.append(time.time() - t)

        # Move the sigma points through the state transition function
        t = time.time()
        u = cupy.asarray(u)
        times.append(time.time() - t)

        t = time.time()
        sigmas += gsf.f_vectorize(sigmas, u, dt)
        times.append(time.time() - t)

        t = time.time()
        sigmas += gsf.state_pdf.draw((gsf.N_particles, gsf._N_sigmas))
        times.append(time.time() - t)

        t = time.time()
        gsf.means = cupy.average(sigmas, axis=1, weights=gsf._w_sigma)
        times.append(time.time() - t)

        t = time.time()
        sigmas -= gsf.means[:, None, :]
        gsf.covariances = sigmas.swapaxes(1, 2) @ (sigmas * gsf._w_sigma[:, None])
        times.append(time.time() - t)

        timess.append(times)

    return numpy.array(timess)
