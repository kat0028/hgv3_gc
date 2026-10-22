from waypoint_traj_solver import *
from hl20_model import HL20Model
import matplotlib.pyplot as plt

model = HL20Model()

def initialState(lat, lon, alt, speed, fpa, heading, alpha=0.0, sigma=0.0):
    veh = HGV3.from_model("hl20", Epoch(2026, 7, 1))
    veh.spawn_from_flight_path(lat=lat, lon=lon, alt=alt, speed=speed, fpa=fpa, heading=heading)
    return stateFromEcef(veh.position_ecef(), veh.velocity_ecef(), alpha, sigma)

if __name__ == "__main__":
    d = np.radians
    model = HL20Model()
    
    x0 = initialState(lat=d(30.0), lon=d(-100.0), alt=60e3, speed=5000.0, fpa=d(-1.0), heading=d(90.0))
    
    waypoints = [
        Waypoint(lat=d(29.371), lon=d(-92.465), alt=53.4e3, radius=1e3),
        Waypoint(lat=d(-28.059), lon=d(-86.093), alt=46.7e3, radius=1e3),
    ]
    
    terminal = TerminalCondition(lat=d(24.434), lon=d(-81.784), alt=44.2e3)
    
    weights = OptimizationWeights(energy=1.0, waypoint=1.0, control=1e-3)
    
    solver = WaypointTrajectorySolver3(model, waypoints, terminal, weights=weights, nodes_per_leg=30, ipopt_options={"print_level": 0})
    
    solver.setInitialState(x0)
    solver.setTargets()
    solver.setInitialGuess()
    res = solver.solve()
    
    n = res.x.shape[1]
    h, mach, qbar, n_load, energy = (np.array(v).ravel() for v in model.outputs.map(n)(res.x))
    lla = np.array([hgv3.ecef_to_lla(np.array(model.ecef(res.x[:, k])).ravel()) for k in range(n)])
    
    plt.figure
    plt.plot(np.degrees(lla[:, 1]), np.degrees(lla[:, 0]))
    
    plt.plot(np.degrees(x0[1]), np.degrees(x0[2]), '*')
    plt.plot(np.degrees(terminal.lon), np.degrees(terminal.lat), '*')
    
    for wp in waypoints:
        plt.plot(np.degrees(wp.lon), np.degrees(wp.lat), 'o')
    
    plt.show()