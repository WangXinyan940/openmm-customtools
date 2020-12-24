class GaMDLangevinIntegrator(mm.CustomIntegrator):
    """
    GaMD Langevin Integrator

    An implementation of Gaussian Accelerated Molecular Dynamics (GaMD). The detail 
    of GaMD can be found here: http://miao.compbio.ku.edu/GaMD.

    This implementation only works on total potential energy. For enhanced sampling
    of specific degrees of freedom, see https://github.com/ljmartin/openmm_gamd. 
    """

    def __init__(self, temperature, friction, dt, sig0=10):
        """Create a GaMDLangevinIntegrator.

        Parameters
        ----------
        temperature
            The simulation temperature to use            
        friction 
            The strength of whitening with random velocity in Langevin dynamics
        dt 
            The integration time step to use
        sig0
            Pre-factor of sigma of dU distribution. sigma_0 = sig0 * kBT
        """
        # convert Units
        self.K = 0.0
        self.E = 10000.0
        if u.is_quantity(temperature):
            temperature = temperature / u.kelvin
        else:
            temperature = temperature
        if u.is_quantity(friction):
            fr = friction
        else:
            fr = friction * u.picosecond
        if u.is_quantity(dt):
            dt = dt
        else:
            dt = dt / u.picosecond
        frdt = fr*dt
        kbT = 8.314 * temperature * 1e-3
        self.sig_0 = 0.008314 * temperature * sig0 # upper limit of sig_dV 
        
        mm.CustomIntegrator.__init__(self, dt)

        # variables for GaMD
        self.addGlobalVariable("fpre", 1)
        self.addGlobalVariable("k", self.K)
        self.addGlobalVariable("E", self.E)

        # variables for Langevin
        self.addGlobalVariable("a", np.exp(-frdt))
        self.addGlobalVariable("b", np.sqrt(1-np.exp(-2*frdt)))
        self.addGlobalVariable("kT", kbT)
        self.addPerDofVariable("x1", 0)

        # integration loop for LangevinMiddleIntegrator
        self.addUpdateContextState()
        self.addComputeGlobal("fpre", "1-k*(E-energy)*step(E-energy)")
        self.addComputePerDof("v", "v + dt*fpre*f/m")
        self.addConstrainVelocities()
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian")
        self.addComputePerDof("x", "x + 0.5*dt*v")
        self.addComputePerDof("x1", "x")
        self.addConstrainPositions()
        self.addComputePerDof("v", "v + (x-x1)/dt")

    def setK(self, newK):
        """Set K value.

        Parameters
        ----------
        newK: 1/energy
            K constant. Use 1/(kJ/mol) as unit if no unit is given
        """
        if u.is_quantity(newK):
            self.K = newK.value_in_unit(u.mol/u.kilojoule)
        else:
            self.K = newK
        self.setGlobalVariableByName("k", newK)

    def getK(self):
        """Get K value.
        """
        return self.getGlobalVariableByName("k")
    
    def setE(self, newE):
        """Set E value.

        Parameters
        ----------
        newE
            E constant. Use kJ/mol as unit if no unit is given
        """
        if u.is_quantity(newE):
            self.E = newE.value_in_unit(u.kilojoule_per_mole)
        else:
            self.E = newE
        self.setGlobalVariableByName("E", newE)

    def getE(self):
        """Get E value.
        """
        return self.getGlobalVariableByName("E")

    def getEffectiveEnergy(self, energy):
        """Get effective energy which obeys
            Eeff = U + k/2*(E-U)^2     if E>U
                 = U                   if E<=U
        """
        return energy + 0.5 * self.K * (self.E - energy) ** 2 if energy < self.E else energy

    def updateParametersBySample(self, energy_list):
        """Update constant K and E based on sampled energy distribution.
        """
        # get statistics
        arr = np.array(energy_list)
        Vmax = arr.max()
        Vmin = arr.min()
        Vavg = arr.mean()
        Vstd = np.std(arr)

        # calc parameters
        E = Vmax
        k_0 = min(1, (self.sig_0/Vstd) * ((Vmax - Vmin) / (Vmax - Vavg)))
        k = k_0 * (1.0 / (Vmax - Vmin))

        self.setK(k)
        self.setE(E)

    def sampleEnergyDistribution(self, context, nsample=200, stepPerSample=100):
        """Sample energies under GaMD effective potential.

        Parameters
        ----------
        context
            OpenMM context binded with this integrator
        nsample = 200
            Number of samples 
        stepPerSample = 100
            Number of steps between two nearest samples
        """
        ener_list = []
        for i in range(nsample):
            self.step(stepPerSample)
            state = context.getState(getEnergy=True)
            ener = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
            ener_list.append(ener)
        
        return np.array(ener_list)