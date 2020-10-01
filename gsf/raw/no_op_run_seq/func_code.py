# first line: 396
@RunSequences.vectorize
@PickleJar.pickle(path='gsf/raw')
def no_op_run_seq(N_time, N_runs):
    """Performs a run sequence on a no-op routine with the given sleep time
     and number of runs

    Parameters
    ----------
    N_time : float
        Sleep time

    N_runs : int
        Number of runs in the sequence

    Returns
    -------
    times : numpy.array
        The times of the run sequence
    """
    times = []

    for _ in tqdm.tqdm(range(N_runs)):
        time.sleep(N_time)
        t = time.time()
        times.append(time.time() - t)

    return numpy.array(times)
