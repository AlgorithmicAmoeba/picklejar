# first line: 157
@PickleJar.pickle(path='pf/processed')
def cpu_gpu_power_seqs():
    """Returns the power sequences for all the runs

    Returns
    -------
    power_seqss : List
        [CPU; GPU] x [predict; update; resample] x [N_particles; power_seq]
    """
    N_particles_cpu = numpy.array([int(i) for i in 2**numpy.arange(0, 24, 0.5)])
    N_particles_gpu = numpy.array([int(i) for i in 2**numpy.arange(0, 24, 0.5)])
    power_seqss = [
        [
            predict_power_seq(N_particles_cpu, 5, False),
            update_power_seq(N_particles_cpu, 5, False),
            resample_power_seq(N_particles_cpu, 5, False)
        ],
        [
            predict_power_seq(N_particles_gpu, 5, True),
            update_power_seq(N_particles_gpu, 5, True),
            resample_power_seq(N_particles_gpu, 5, True)
        ]
    ]
    
    for cpu_gpu in range(2):
        for method in range(3):
            _, powers = power_seqss[cpu_gpu][method]
            for i, (N_runs, power) in enumerate(powers):
                powers[i] = power / N_runs
        
    return power_seqss
