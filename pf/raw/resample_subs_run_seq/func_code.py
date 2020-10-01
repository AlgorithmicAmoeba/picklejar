# first line: 238
@RunSequences.vectorize
@PickleJar.pickle(path='pf/raw')
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

    _, _, _, pf = sim_base.get_parts(
        N_particles=N_particle,
        gpu=True,
        pf=True
    )

    for _ in tqdm.tqdm(range(N_runs)):
        u, _ = sim_base.get_random_io()
        pf.predict(u, 1.)

        times = []

        t = time.time()
        t_weights = torch_dlpack.from_dlpack(cupy.asarray(pf.weights).toDlpack())
        t_cumsum = torch.cumsum(t_weights, 0)
        cumsum = cupy.fromDlpack(torch_dlpack.to_dlpack(t_cumsum))
        cumsum /= cumsum[-1]
        times.append(time.time() - t)

        t = time.time()
        sample_index = cupy.zeros(pf.N_particles, dtype=cupy.int64)
        random_number = cupy.float64(cupy.random.rand())

        filter.particle.ParallelParticleFilter._parallel_resample[pf._bpg, pf._tpb](
            cumsum, sample_index,
            random_number,
            pf.N_particles
        )
        times.append(time.time() - t)

        t = time.time()
        pf.particles = cupy.asarray(pf.particles)[sample_index]
        pf.weights = cupy.full(pf.N_particles, 1 / pf.N_particles)
        times.append(time.time() - t)

        timess.append(times)

    return numpy.array(timess)
