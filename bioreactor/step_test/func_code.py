# first line: 12
@PickleJar.pickle(path='bioreactor')
def step_test(percent, dt):
    # Simulation set-up
    end_time = 300
    ts = numpy.linspace(0, end_time, int(end_time//dt))

    bioreactor, lin_model, _, _ = sim_base.get_parts()

    # Initial values
    u = numpy.array([0.06, 0.2])
    u *= percent
    ys = [bioreactor.outputs(u)]

    for _ in tqdm.tqdm(ts[1:]):

        bioreactor.step(dt, u)
        outputs = bioreactor.outputs(u)
        ys.append(outputs.copy())

    ys = numpy.array(ys)
    return ts, ys
