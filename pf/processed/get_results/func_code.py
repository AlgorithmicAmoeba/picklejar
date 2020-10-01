# first line: 26
@PickleJar.pickle(path='pf/processed')
def get_results(end_time=50, monte_carlo_sims=1):
    run_seqss = PF_run_seq.cpu_gpu_run_seqs()
    powerss = PF_power.cpu_gpu_power_seqs()

    energy_cpugpu, runtime_cpugpu, mpc_frac_cpugpu, performance_cpugpu, pcov_cpugpu = [], [], [], [], []
    for cpu_gpu in range(2):
        sums = numpy.min(run_seqss[cpu_gpu][0][1], axis=1)
        N_particles = run_seqss[0][0][0]

        for method in range(1, 3):
            _, run_seqs = run_seqss[cpu_gpu][method]
            times = numpy.min(run_seqs, axis=1)
            sums += times

        runtime_cpugpu.append(sums)

        method_power = []
        for method in range(3):
            _, powers = powerss[cpu_gpu][method]
            power = powers[:, 0]
            if cpu_gpu:
                power += powers[:, 1]

            method_power.append(power)

        energyss, mpc_fracss, performancess, pcovss = [], [], [], []
        for i in range(len(N_particles)):
            dt_control = dt_predict = 0.1

            energys, mpc_fracs, performances, pcovs = [], [], [], []
            for monte_carlo in range(monte_carlo_sims):
                performance, mpc_frac, predict_count, update_count, pcov = get_sim(
                    int(N_particles[i]),
                    dt_control,
                    dt_predict,
                    monte_carlo,
                    pf=True,
                    end_time=end_time
                )

                predict_energy, update_energy, resample_energy = [method_power[j][i] for j in range(3)]
                total_energy = predict_count * predict_energy + update_count * (update_energy + resample_energy)

                energys.append(total_energy)
                mpc_fracs.append(mpc_frac)
                performances.append(performance)
                pcovs.append(pcov)

            energyss.append(energys)
            mpc_fracss.append(mpc_fracs)
            performancess.append(performances)
            pcovss.append(pcovs)

        energy_cpugpu.append(energyss)
        mpc_frac_cpugpu.append(mpc_fracss)
        performance_cpugpu.append(performancess)
        pcov_cpugpu.append(pcovss)

    N_particles = [run_seqss[0][0][0], run_seqss[0][0][0]]
    energy_cpugpu = numpy.array(energy_cpugpu)
    mpc_frac_cpugpu = numpy.array(mpc_frac_cpugpu)
    performance_cpugpu = numpy.array(performance_cpugpu)
    pcov_cpugpu = numpy.array(pcov_cpugpu)

    return N_particles, energy_cpugpu, runtime_cpugpu, mpc_frac_cpugpu, performance_cpugpu, pcov_cpugpu