#!/usr/bin/env python3
"""Stand-in HL-20 3-DOF model for WaypointTrajectorySolver3.

Point mass over a rotating WGS84 Earth with J2 gravity, aero and atmosphere fit
from the installed hgv3 package. Matched hgv3 open-loop to ~10 m in altitude
over 800 s; position drifts ~0.8 m/s along-track because hgv3 co-rotates the
atmosphere about ECI z rather than the ECEF pole.

States:   [r, lon, lat, V, gamma, psi, alpha, sigma]
          geocentric r [m], lon/lat [rad]; Earth-relative V [m/s], gamma, psi [rad]
Controls: [alpha_dot, sigma_dot] [rad/s]
"""

import casadi as ca
import numpy as np

import hgv3
from waypoint_traj_solver import WGS84_A, aeroFits, alphaLimitFits, atmosphereFits, geodeticAltitude


class HL20Model:
    nx = 8
    nu = 2

    def __init__(self, model_name="hl20", bank_max_deg=85.0, alpha_rate_deg=2.0, bank_rate_deg=10.0):
        af = hgv3.HGV3.load_airframe(model_name)
        self.mass = af.mass
        self.s_ref = af.ref.s_ref
        self.CL, self.CD = aeroFits(af)
        self.alpha_lo, self.alpha_hi = alphaLimitFits(af.aero)
        self.log_rho, self.sound_speed = atmosphereFits()

        d = np.radians
        rows = af.aero.rows
        a_lo = min(r.alpha_min for r in rows)
        a_hi = max(r.alpha_max for r in rows)
        # alpha box spans the whole table; alpha_limits does the Mach-dependent limiting
        self.x_lb = np.array([WGS84_A - 5e3, -np.inf, d(-89), 50.0, d(-80), -np.inf, d(a_lo), -d(bank_max_deg)])
        self.x_ub = np.array([WGS84_A + 200e3, np.inf, d(89), 9e3, d(80), np.inf, d(a_hi), d(bank_max_deg)])
        # rate limits are placeholders, not HL-20 values
        self.u_ub = np.array([d(alpha_rate_deg), d(bank_rate_deg)])
        self.u_lb = -self.u_ub

        self.x_scale = np.array([1e4, 0.1, 0.1, 1e3, 0.1, 1.0, 0.1, 1.0])
        self.x_offset = np.array([WGS84_A, 0, 0, 0, 0, 0, 0, 0])
        self.u_scale = self.u_ub.copy()

        x = ca.MX.sym("x", self.nx)
        u = ca.MX.sym("u", self.nu)
        mach = ca.MX.sym("mach")
        out = self.outputExprs(x)
        self.f = ca.Function("f", [x, u], [self.xdot(x, u)], ["x", "u"], ["xdot"])
        self.outputs = ca.Function("outputs", [x], list(out.values()), ["x"], list(out))
        self.ecef = ca.Function("ecef", [x], [self.ecefExpr(x)], ["x"], ["p"])
        self.alpha_limits = ca.Function("alpha_limits", [mach], [self.alpha_lo(mach), self.alpha_hi(mach)],
                                        ["mach"], ["lo", "hi"])

    def aero(self, x):
        r, lon, lat, V, gam, psi, alpha, sigma = ca.vertsplit(x)
        h = geodeticAltitude(r, lat)
        rho = ca.exp(self.log_rho(h))
        mach = V / self.sound_speed(h)
        qbar = 0.5 * rho * V**2
        L = qbar * self.s_ref * self.CL(ca.vertcat(alpha, mach)) / self.mass     # per unit mass
        D = qbar * self.s_ref * self.CD(ca.vertcat(alpha, mach)) / self.mass
        return h, mach, qbar, L, D

    def xdot(self, x, u):
        r, lon, lat, V, gam, psi, alpha, sigma = ca.vertsplit(x)
        _, _, _, L, D = self.aero(x)
        mu, J2, Re, Om = hgv3.MU_EARTH, hgv3.J2, hgv3.R_EARTH, hgv3.OMEGA_EARTH

        sl, cl = ca.sin(lat), ca.cos(lat)
        sg, cg = ca.sin(gam), ca.cos(gam)
        sp, cp = ca.sin(psi), ca.cos(psi)

        # point mass + J2: radial (down) magnitude and northward component
        g_r = mu / r**2 * (1 - 1.5 * J2 * (Re / r)**2 * (3 * sl**2 - 1))
        g_n = -3 * mu * J2 * Re**2 * sl * cl / r**4

        r_dot = V * sg
        lon_dot = V * cg * sp / (r * cl)
        lat_dot = V * cg * cp / r
        V_dot = (-D - g_r * sg + g_n * cg * cp
                 + Om**2 * r * cl * (sg * cl - cg * sl * cp))
        gam_dot = ((L * ca.cos(sigma) - g_r * cg - g_n * sg * cp) / V + V * cg / r
                   + 2 * Om * cl * sp
                   + Om**2 * r * cl * (cg * cl + sg * sl * cp) / V)
        psi_dot = ((L * ca.sin(sigma) - g_n * sp) / (V * cg)
                   + V * cg * sp * ca.tan(lat) / r
                   - 2 * Om * (cl * cp * ca.tan(gam) - sl)
                   + Om**2 * r * sl * cl * sp / (V * cg))
        return ca.vertcat(r_dot, lon_dot, lat_dot, V_dot, gam_dot, psi_dot, u[0], u[1])

    def outputExprs(self, x):
        h, mach, qbar, L, D = self.aero(x)
        return {"h": h,
                "mach": mach,
                "qbar": qbar,
                "n_load": ca.sqrt(L**2 + D**2) / hgv3.G0,
                "energy": 0.5 * x[3]**2 + hgv3.G0 * h}

    @staticmethod
    def ecefExpr(x):
        r, lon, lat = x[0], x[1], x[2]
        return r * ca.vertcat(ca.cos(lat) * ca.cos(lon), ca.cos(lat) * ca.sin(lon), ca.sin(lat))
