# first line: 242
@RunSequences.vectorize
@PickleJar.pickle(path='gsf/raw')
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

    _, _, _, gsf = sim_base.get_parts(
        N_particles=N_particle,
        gpu=True,
        pf=False
    )

    for _ in tqdm.tqdm(range(N_runs)):
        u, z = sim_base.get_random_io()
        gsf.predict(u, 1.)

        times = []
        t = time.time()
        # Local Update
        sigmas = gsf._get_sigma_points()
        times.append(time.time() - t)

        # Move the sigma points through the state observation function
        t = time.time()
        u = cupy.asarray(u)
        times.append(time.time() - t)

        t = time.time()
        etas = gsf.g_vectorize(sigmas, u, gsf._y_dummy)
        times.append(time.time() - t)

        # Compute the Kalman gain
        t = time.time()
        eta_means = cupy.average(etas, axis=1, weights=gsf._w_sigma)
        sigmas -= gsf.means[:, None, :]
        etas -= eta_means[:, None, :]

        P_xys = sigmas.swapaxes(1, 2) @ (etas * gsf._w_sigma[:, None])
        P_yys = etas.swapaxes(1, 2) @ (etas * gsf._w_sigma[:, None])
        P_yy_invs = cupy.linalg.inv(P_yys)
        Ks = P_xys @ P_yy_invs
        times.append(time.time() - t)

        # Use the gain to update the means and covariances
        t = time.time()
        z = cupy.asarray(z, dtype=cupy.float32)
        times.append(time.time() - t)

        t = time.time()
        es = z - eta_means
        gsf.means += (Ks @ es[:, :, None]).squeeze()
        # Dimensions from paper do not work, use corrected version
        gsf.covariances -= Ks @ P_yys @ Ks.swapaxes(1, 2)
        times.append(time.time() - t)

        # Global Update
        # Move the means through the state observation function
        t = time.time()
        y_means = gsf.g_vectorize(gsf.means, u, gsf._y_dummy)
        times.append(time.time() - t)

        t = time.time()
        glob_es = z - y_means
        gsf.weights *= gsf.measurement_pdf.pdf(glob_es)
        times.append(time.time() - t)

        timess.append(times)

    return numpy.array(timess)
