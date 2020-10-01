# first line: 18
@PickleJar.pickle(path='gsf/raw')
def get_sim(N_particles, dt_control, dt_predict, monte_carlo=0, end_time=50, pf=False):
    _ = monte_carlo
    sim = sim_base.Simulation(N_particles, dt_control, dt_predict, end_time, pf)
    sim.simulate()
    ans = sim.performance, sim.mpc_frac, sim.predict_count, sim.update_count, sim.covariance_point_size
    return ans
