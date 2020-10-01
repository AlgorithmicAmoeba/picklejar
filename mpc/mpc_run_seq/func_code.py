# first line: 11
@PickleJar.pickle(path='mpc')
def mpc_run_seq(N_runs):
    """Performs a run sequence on the MPC step function for a
     number of runs

    Parameters
    ----------
    N_runs : int
        Number of runs in the sequence

    Returns
    -------
    times : numpy.array
        The times of the run sequence
    """

    times = []

    bioreactor, lin_model, K, _ = sim_base.get_parts(dt_control=0.1)

    state_pdf, measurement_pdf = sim_base.get_noise()

    # Initial values
    dt = 0.1
    us = [numpy.array([0.06, 0.2])]
    xs = [bioreactor.X.copy()]
    ys = [bioreactor.outputs(us[-1])]
    ys_meas = [bioreactor.outputs(us[-1])]

    biass = []

    for _ in tqdm.tqdm(range(N_runs)):
        U_temp = us[-1].copy()
        if K.y_predicted is not None:
            biass.append(lin_model.yn2d(ys_meas[-1]) - K.y_predicted)

        t = time.time()
        # noinspection PyBroadException
        try:
            u = K.step(
                lin_model.xn2d(xs[-1]),
                lin_model.un2d(us[-1]),
                lin_model.yn2d(ys_meas[-1])
            )
        except:
            u = numpy.array([0.06, 0.2])
        times.append(time.time() - t)

        U_temp[lin_model.inputs] = lin_model.ud2n(u)
        us.append(U_temp.copy())

        bioreactor.step(dt, us[-1])
        bioreactor.X += state_pdf.draw().get().squeeze()
        outputs = bioreactor.outputs(us[-1])
        ys.append(outputs.copy())
        outputs[lin_model.outputs] += measurement_pdf.draw().get().squeeze()
        ys_meas.append(outputs)
        xs.append(bioreactor.X.copy())

    return numpy.array(times)
