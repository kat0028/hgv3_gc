#!/usr/bin/env python3

from dataclasses import dataclass
from typing import Optional

import casadi as ca 
import numpy as np

import hgv3
from hgv3 import HGV3, Epoch

WGS84_A = hgv3.WGS84_A
WGS84_B = hgv3.WGS84_A * (1.0 - hgv3.WGS84_F)

def aeroFits(airframe, d_alpha_deg=0.5, mach_max=30.0):
    rows = airframe.aero.rows
    a_lo = min(r.alpha_min for r in rows)
    a_hi = max(r.alpha_max for r in rows)
    
    alphas = np.radians(np.arange(a_lo, a_hi + 1e-9, d_alpha_deg))
    machs = np.unique(np.r_[np.linspace(rows[0].mach, rows[-1].mach, 120),
                            np.linspace(rows[-1].mach, mach_max, 20), 
                            [r.mach for r in rows]])
    CL = np.empty((alphas.size, machs.size))
    CD = np.empty_like(CL)
    
    for i, a in enumerate(alphas):
        for j, M in enumerate(machs):
            c = hgv3.wind_coefficients(a, M, airframe)
            CL[i, j], CD[i, j] = c.cl, c.cd
    
    grid = [alphas, machs]
    return (ca.interpolant("CL", "bspline", grid, CL.ravel(order="F")),
            ca.interpolant("CD", "bspline", grid, CD.ravel(order="F")))
    
def alphaLimitFits(db, ramp=1.0):
    rows = db.rows
    
    lo = [rows[0].alpha_min] + [max(a.alpha_min, b.alpha_min) for a, b in zip(rows, rows[1:])] + [rows[-1].alpha_min]
    hi = [rows[0].alpha_max] + [min(a.alpha_max, b.alpha_max) for a, b in zip(rows, rows[1:])] + [rows[-1].alpha_max]  
    
    machs = [r.mach for r in rows]
    lo_k = [max(lo[k], lo[k + 1]) for k in range(len(rows))]
    hi_k = [min(hi[k], hi[k + 1]) for k in range(len(rows))]
    
    grid = [0.0] + machs + [machs[-1] + ramp, 100.0]
    lo_k = [lo[0]] + lo_k + [lo[-1], lo[-1]]
    hi_k = [hi[0]] + hi_k + [hi[-1], hi[-1]]
    
    opts = {"degree": [1]}
    return (ca.interpolant("alpha_lo", "bspline", [grid], np.radians(lo_k), opts),
            ca.interpolant("alpha_hi", "bspline", [grid], np.radians(hi_k), opts))

def atmosphereFits(h_min = -2e3, h_max = 150e3, dh=500.0):
    h = np.arange(h_min, h_max + 1.0, dh)
    atm = [hgv3.atmosphere(z) for z in h]
    return (ca.interpolant("log_rho", "bspline", [h], np.log([s.density for s in atm])),
            ca.interpolant("sound_speed", "bspline", [h], [s.sound_speed for s in atm]))
    
def geodeticAltitude(r, lat):
    return r - WGS84_A * WGS84_B / ca.sqrt((WGS84_B * ca.cos(lat))**2 + (WGS84_A * ca.sin(lat))**2)

def ellipsoidRadius(lat):
    return WGS84_A * WGS84_B / np.sqrt((WGS84_B * np.cos(lat))**2 + (WGS84_A * np.sin(lat))**2)


def stateFromEcef(p, v, alpha, sigma):
    p, v = np.asarray(p, float), np.asarray(v, float)
    r = np.linalg.norm(p)
    lon, lat = np.arctan2(p[1], p[0]), np.arcsin(p[2] / r)
    up = p / r
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.cross(up, east)
    V = np.linalg.norm(v)
    return np.array([r, lon, lat, V, np.arcsin(v @ up / V), np.arctan2(v @ east, v @ north), alpha, sigma])

@dataclass 
class Waypoint: 
    lat: float
    lon: float 
    alt: float
    radius: float = 1e3
    heading: Optional[float] = None
    heading_tol: float = np.radians(2.0)
    
@dataclass 
class TrajectoryResult:
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    leg_times: np.ndarray
    success: bool
    stats: dict

@dataclass 
class OptimizationWeights:
    control: float = 1e-3
    waypoint: float = 1.0
    energy: float = 1.0
    
@dataclass
class TerminalCondition:
    lat: float
    lon: float
    alt: float
    speed: Optional[float] = None
    flight_path_angle: Optional[float] = None
    heading: Optional[float] = None

class WaypointTrajectorySolver3:
    def __init__(self, model, waypoints, terminal, weights=None, nodes_per_leg=30, rk_substeps=2,
                 leg_time_bounds=(10.0, 3000.0), h_min=5e3, qbar_max=40e3, n_load_max=3.0,
                 ipopt_options=None):
        self.model = model
        self.waypoints = list(waypoints)
        self.terminal = terminal
        self.weights = weights or OptimizationWeights()
        self.n_legs = len(self.waypoints) + 1             # last leg ends at the terminal point
        self.N = nodes_per_leg
        self.K = self.n_legs * self.N
        self.wp_nodes = [(j + 1) * self.N for j in range(len(self.waypoints))]

        self.opti = ca.Opti()
        self.createVariables(leg_time_bounds)
        self.addDynamics(rk_substeps)
        self.addInitialConditions()
        self.addTerminalConditions()
        self.addWaypointCost()
        self.addPathConstraints(h_min, qbar_max, n_load_max)
        self.setObjective()
        self.configureSolver(ipopt_options)
        
    def createVariables(self, leg_time_bounds):
        m = self.model
        opti = self.opti
        self.X = opti.variable(m.nx, self.K + 1)
        self.U = opti.variable(m.nu, self.K)
        self.T = opti.variable(self.n_legs)
        
        self.x0 = opti.parameter(6)
        self.xf = opti.parameter(6)
        self.wp_ecef = opti.parameter(3, len(self.waypoints))
        
        rep = lambda v, n: ca.repmat(ca.DM(v), 1, n)
        opti.set_linear_scale(self.X, rep(m.x_scale, self.K + 1), rep(m.x_offset, self.K + 1))
        opti.set_linear_scale(self.U, rep(m.u_scale, self.K))
        opti.set_linear_scale(self.T, 100.0)
        
        for i in range(m.nx):
            if np.isfinite(m.x_lb[i]):
                opti.subject_to(opti.bounded(m.x_lb[i], self.X[i, :], m.x_ub[i]))
                
        opti.subject_to(opti.bounded(rep(m.u_lb, self.K), self.U, rep(m.u_ub, self.K)))
        opti.subject_to(opti.bounded(leg_time_bounds[0], self.T, leg_time_bounds[1]))
        
    def rk4Step(self, substeps):
        x = ca.MX.sym("x", self.model.nx)
        u = ca.MX.sym("u", self.model.nu)
        dt = ca.MX.sym("dt")
        
        f = self.model.f 
        h = dt / substeps
        xk = x
        
        for _ in range(substeps):
            k1 = f(xk, u)
            k2 = f(xk + h / 2 * k1, u)
            k3 = f(xk + h / 2 * k2, u)
            k4 = f(xk + h * k3, u)
            xk = xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        
        return ca.Function("F", [x, u, dt], [xk])
    
    def addDynamics(self, rk_substeps):
        F = self.rk4Step(rk_substeps).map(self.N)
        scale = ca.repmat(ca.DM(self.model.x_scale), 1, self.N)
        for j in range(self.n_legs):
            k0 = j * self.N
            k1 = (j+1) * self.N
            dt = ca.repmat(self.T[j] / self.N, 1, self.N)
            x_next = F(self.X[:, k0:k1], self.U[:, k0:k1], dt)
            self.opti.subject_to((self.X[:, k0+1:k1+1] - x_next) / scale == 0)
            
    def addInitialConditions(self):
        self.opti.subject_to(self.X[:6, 0] == self.x0)
        
    def addTerminalConditions(self):
        tc = self.terminal
        fixed = [0, 1, 2] + [i for i, v in ((3, tc.speed), (4, tc.flight_path_angle), (5, tc.heading)) if v is not None]
        for i in fixed: 
            self.opti.subject_to((self.X[i, -1] - self.xf[i]) / self.model.x_scale[i] == 0)
            
    def addWaypointCost(self):
        self.wp_cost = 0
        for j, (wp, k) in enumerate(zip(self.waypoints, self.wp_nodes)):
            miss = self.model.ecef(self.X[:, k]) - self.wp_ecef[:, j]
            self.wp_cost += ca.sumsqr(miss / wp.radius)
            if wp.heading is not None:
                self.wp_cost += (1 - ca.cos(self.X[5, k] - wp.heading)) / (1 - np.cos(wp.heading_tol))
                
    def addPathConstraints(self, h_min, qbar_max, n_load_max):
        self.h, self.mach, self.qbar, self.n_load, self.energy = self.model.outputs.map(self.K + 1)(self.X)
        self.opti.subject_to(self.h / 1e4 >= h_min / 1e4)
        self.opti.subject_to(self.qbar / 1e4 <= qbar_max / 1e4)
        self.opti.subject_to(self.n_load <= n_load_max)
        
        a_lo, a_hi = self.model.alpha_limits.map(self.K + 1)(self.mach)
        self.opti.subject_to(self.X[6, :] >= a_lo)
        self.opti.subject_to(self.X[6, :] <= a_hi)
        
    def setObjective(self):
        w = self.weights
        J = -w.energy * self.energy[-1] / 1e7
        J += w.waypoint * self.wp_cost
        u_s = self.U / ca.repmat(ca.DM(self.model.u_scale), 1, self.K)
        J += w.control * ca.sumsqr(u_s) / self.K
        self.opti.minimize(J)
        
    def configureSolver(self, ipopt_options):
        opts = {"max_iter": 3000, "tol": 1e-6, "mu_strategy": "adaptive", "print_level": 5}
        opts.update(ipopt_options or {})
        self.opti.solver("ipopt", {"expand": True}, opts)
        
    # Per Solve 
    def setInitialState(self, x0):
        self._x0 = np.asarray(x0, float)
        self.opti.set_value(self.x0, self._x0[:6])
        
    def setTargets(self):
        if self.waypoints:
            self.opti.set_value(self.wp_ecef, np.column_stack([hgv3.lla_to_ecef(w.lat, w.lon, w.alt) for w in self.waypoints]))
        
        tc = self.terminal
        x0 = self._x0
        p = hgv3.lla_to_ecef(tc.lat, tc.lon, tc.alt)
        r = np.linalg.norm(p)
        lon, lat = np.arctan2(p[1], p[0]), np.arcsin(p[2] / r)
        
        near = lambda a, ref: ref + (a - ref + np.pi) % (2 * np.pi) - np.pi
        
        xf = [r, near(lon, x0[1]), lat, 
              tc.speed if tc.speed is not None else 0.0,
              tc.flight_path_angle if tc.flight_path_angle is not None else 0.0,
              near(tc.heading, x0[5]) if tc.heading is not None else 0.0]
        self.opti.set_value(self.xf, xf)
        
    def setInitialGuess(self, V_final_frac=0.5, alpha=np.radians(25.0)):
        x0 = self._x0
        tc = self.terminal
        pts = np.array([(x0[2], x0[1], x0[0] - ellipsoidRadius(x0[2]))] + [(w.lat, w.lon, w.alt) for w in self.waypoints + [tc]])
        
        s = np.linspace(0.0, 1.0, self.N + 1)[:-1, None]
        llh = np.vstack([a + s * (b-a) for a, b in zip(pts[:-1], pts[1:])] + [pts[-1:]])
        lat, lon, alt = llh.T
        r = alt + ellipsoidRadius(lat)
        V_end = tc.speed if tc.speed is not None else V_final_frac * x0[3]
        V = np.linspace(x0[3], V_end, self.K + 1)
        
        dn = np.diff(lat) * r[:-1]
        de = np.diff(lon) * r[:-1] * np.cos(lat[:-1])
        ds = np.hypot(dn, de)
        psi = np.unwrap(np.append(np.arctan2(de, dn), np.arctan2(de[-1], dn[-1])))
        gam = np.arctan2(np.diff(alt), ds)
        gam = np.append(gam, gam[-1])
        
        dt = ds/(0.5 * (V[:-1] + V[1:]))
        T = dt.reshape(self.n_legs, self.N).sum(axis=1)
        
        X = np.vstack([r, lon, lat, V, gam, psi, np.full(self.K + 1, alpha), np.zeros(self.K + 1)])
        
        self.opti.set_initial(self.X, X)
        self.opti.set_initial(self.U, 0)
        self.opti.set_initial(self.T, T)
        
    def warmStart(self, result):
        self.opti.set_initial(self.X, result.x)
        self.opti.set_initial(self.U, result.u)
        self.opti.set_initial(self.T, result.leg_times)
        
    def solve(self):
        try: 
            sol = self.opti.solve()
            ok, val = True, sol.value
        except RuntimeError:
            ok, val = False, self.opti.debug.value
        T = np.atleast_1d(val(self.T))
        t = np.concatenate([[0.0], np.cumsum(np.repeat(T / self.N, self.N))])
        return TrajectoryResult(t=t, x=val(self.X), u=val(self.U), leg_times=T, success=ok, stats=self.opti.stats())