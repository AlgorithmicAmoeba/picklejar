# first line: 78
@PickleJar.pickle(path='bioreactor/perf_vs_cp/processed')
def generate_results():
    """Collects individual simulation results for performance runs

    Returns
    -------
    dt_controls, performances : list
        A list of control periods and performances
    """
    monte_carlos = 5
    dt_controls, performances = [], []
    for dt_control in tqdm.tqdm(numpy.linspace(0.1, 30, 20)):
        for monte_carlo in range(monte_carlos):
            y = get_simulation_performance(dt_control, monte_carlo)
            if y > 1e8:
                continue
            dt_controls.append(dt_control)
            performances.append(y)

    return dt_controls, performances
