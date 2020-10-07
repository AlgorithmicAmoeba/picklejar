# first line: 9
@PickleJar.pickle(path='bioreactor/perf_vs_cp/raw')
def get_simulation_performance(dt_control, monte_carlo):
    """Does a simulation with a given control period and returns the performance

    Parameters
    ----------
    dt_control : float
        Control period

    monte_carlo : int
        Index of the monte carlo run

    Returns
    -------
    performance : float
        ISE performance of the run

    """
    _ = monte_carlo
    end_time = 50
    ts = numpy.linspace(0, end_time, end_time*20)
    dt = ts[1]
    assert dt <= dt_control

    bioreactor, lin_model, K, _ = sim_base.get_parts(dt_control=dt_control)
    state_pdf, measurement_pdf = sim_base.get_noise()

    # Initial values
    us = [numpy.array([0.06, 0.2])]
    xs = [bioreactor.X.copy()]
    ys = [bioreactor.outputs(us[-1])]

    biass = []

    t_next = 0
    for t in ts[1:]:
        if t > t_next:
            # noinspection PyUnresolvedReferences
            U_temp = us[-1].copy()
            if K.y_predicted is not None:
                biass.append(lin_model.yn2d(ys[-1]) - K.y_predicted)

            # noinspection PyBroadException
            try:
                u = K.step(
                    lin_model.xn2d(xs[-1]),
                    lin_model.un2d(us[-1]),
                    lin_model.yn2d(ys[-1])
                )
            except:
                u = numpy.array([0.06, 0.2])
            U_temp[lin_model.inputs] = lin_model.ud2n(u)
            us.append(U_temp.copy())
            t_next += dt_control
        else:
            us.append(us[-1])

        bioreactor.step(dt, us[-1])
        bioreactor.X += state_pdf.draw().get().squeeze()
        outputs = bioreactor.outputs(us[-1])
        outputs[lin_model.outputs] += measurement_pdf.draw().get().squeeze()
        ys.append(outputs.copy())
        xs.append(bioreactor.X.copy())

    ys = numpy.array(ys)

    return sim_base.performance(ys[:, lin_model.outputs], lin_model.yd2n(K.ysp), ts)
