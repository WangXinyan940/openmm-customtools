class ITSLangevinIntegrator(CustomIntegrator):
    """
    """

    def __init__(self, dt, friction, nlist, tlist):
        """Create an ITS Langevin integrator.
        The whole integration on temperature would be compiled in integrator kernel
        after initialized. So the parameters cannot be hot-updated during simulation.
        """
        if u.is_quantity(friction):
            fr = friction
        else:
            fr = friction * u.picosecond
        if u.is_quantity(dt):
            dt = dt
        else:
            dt = dt / u.picosecond
        frdt = fr*dt
        temperature = min(tlist)
        mm.CustomIntegrator.__init__(self, dt)
        # generate Eeff expression
        Wupper, Wlower = [], []
        assert len(nlist) == len(tlist)
        for i in range(len(nlist)):
            n = nlist[i]
            t = tlist[i]
            if u.is_quantity(t):
                t = t / u.kelvin
            betap = 1. / (8.314e-3 * t)
            Wupper.append("%.8f*%.8f*exp(-%.8f*energy+1/kT*energy)"%(n, betap, betap))
            Wlower.append("%.8f*exp(-%.8f*energy+1/kT*energy)"%(n, betap))
        self.Wupper = "+".join(Wupper)
        self.Wlower = "+".join(Wlower)

        # Langevin params
        self.addGlobalVariable("a", np.exp(-frdt))
        self.addGlobalVariable("b", np.sqrt(1-np.exp(-2*frdt)))
        self.addGlobalVariable("kT", 8.314e-3*temperature)
        self.addPerDofVariable("x1", 0)

        # ITS params
        self.addGlobalVariable("Wupper", 0)
        self.addGlobalVariable("Wlower", 0)
        self.addGlobalVariable("fprime", 1)

        # Langevin loop
        self.addUpdateContextState()
        self.addComputeGlobal("Wupper", self.Wupper_txt)
        self.addComputeGlobal("Wlower", self.Wlower_txt)
        self.addComputeGlobal("fprime", "Wupper/Wlower*kT")
        self.addComputePerDof("v", "v + dt*fprime*f/m")
        self.addConstrainVelocities()
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian")
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + (x-x1)/dt")


def generateITSParameters(t, Tlist, Elist, Tlow=298.15, Tup=373.15):
    """Generate n and T for ITS simulation.

    Parameters
    ----------
    t
        Exchange prob
    Tlist
        List of temperature for fitting
    Elist
        List of average potential energy for fitting
    Tlow
        The lower limit temperature in ITS temperature integration
    Tup
        The upper limit temperature in ITS temperature integration
    """
    pass