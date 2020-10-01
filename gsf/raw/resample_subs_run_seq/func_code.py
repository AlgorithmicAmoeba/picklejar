# first line: 329
@RunSequences.vectorize
@PickleJar.pickle(path='gsf/raw')
def resample_subs_run_seq(N_particle, N_runs):
    """Performs a run sequence on the resample function's subroutines
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
        t_weights = torch_dlpack.from_dlpack(cupy.asarray(gsf.weights).toDlpack())
        t_cumsum = torch.cumsum(t_weights, 0)
        cumsum = cupy.fromDlpack(torch_dlpack.to_dlpack(t_cumsum))
        cumsum /= cumsum[-1]
        times.append(time.time() - t)

        t = time.time()
        sample_index = cupy.zeros(gsf.N_particles, dtype=cupy.int64)
        random_number = cupy.float64(cupy.random.rand())

        if gsf.N_particles >= 1024:
            threads_per_block = 1024
            blocks_per_grid = (gsf.N_particles - 1) // threads_per_block + 1
        else:
            div_32 = (gsf.N_particles - 1) // 32 + 1
            threads_per_block = 32 * div_32
            blocks_per_grid = 1

        filter.gs_ukf.ParallelGaussianSumUnscentedKalmanFilter._parallel_resample[blocks_per_grid, threads_per_block](
            cumsum, sample_index, random_number, gsf.N_particles
        )
        times.append(time.time() - t)

        t = time.time()
        gsf.means = cupy.asarray(gsf.means)[sample_index]
        gsf.covariances = cupy.asarray(gsf.covariances)[sample_index]
        gsf.weights = cupy.full(gsf.N_particles, 1 / gsf.N_particles)
        times.append(time.time() - t)

        timess.append(times)

    return numpy.array(timess)
