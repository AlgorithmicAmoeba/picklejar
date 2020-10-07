# first line: 19
@PickleJar.pickle(path='pf/raw')
def get_sim(N_particles, dt_control, dt_predict, monte_carlo=0, end_time=50, pf=True):
    """Returns simulations results for a given simulation configuration.

    Parameters
    ----------
    N_particles : int
        Number of particles

    dt_control, dt_predict : float
        Control and prediction periods

    monte_carlo : int, optional
        The monte carlo indexing number

    end_time : float, optional
        Simulation end time

    pf : bool, optional
        Should the filter be the particle filter or gaussian sum filter

    Returns
    -------
    performance : float
        The simulation's ISE performance

    mpc_frac : float
        Fraction of MPC convergence

    predict_count, update_count : int
        Number of times the predict and update/resample methods were called

    covariance_point_size : numpy.array
        Maximum singular value of the covariance point estimate for each time instance
    """
    _ = monte_carlo
    sim = sim_base.Simulation(N_particles, dt_control, dt_predict, end_time, pf)
    sim.simulate()
    ans = sim.performance, sim.mpc_frac, sim.predict_count, sim.update_count, sim.covariance_point_size
    return ans
