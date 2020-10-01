# first line: 501
@PickleJar.pickle(path='gsf/processed')
def gsf_sub_routine_run_seqs():
    """Returns the run sequences for the predict, update and resample subroutines

    Returns
    -------
    run_seqss : List
        [predict; update; resample] x [N_particles; run_seq]
    """
    N_particles_gpu = numpy.array([int(i) for i in 2**numpy.arange(1, 19, 0.5)])
    run_seqss = [
        predict_subs_run_seq(N_particles_gpu, 100),
        update_subs_run_seq(N_particles_gpu, 100),
        resample_subs_run_seq(N_particles_gpu, 100)
    ]
    return run_seqss
