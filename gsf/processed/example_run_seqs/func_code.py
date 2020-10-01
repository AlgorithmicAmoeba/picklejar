# first line: 456
@PickleJar.pickle(path='gsf/processed')
def example_run_seqs():
    """Returns the run sequences for the no_op and time.time() methods

    Returns
    -------
    run_seqss : List
        [no_op; time_time] x [N_particles; run_seq]
    """
    N_times = numpy.array([0.])
    run_seqss = [
        no_op_run_seq(N_times, 100),
        time_time_run_seq(N_times, 100)
    ]
    return run_seqss
