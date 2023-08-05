using Molly

n_atoms = 100
boundary = CubicBoundary(2.0u"nm")  # from -1 to 1
temp = 298.0u"K"
atom_mass = 1.0u"u"

atoms = [Atom(mass=atom_mass, σ=0.3u"nm", ϵ=0.2u"kJ * mol^-1") for i in 1:n_atoms]
coords = place_atoms(n_atoms, boundary; min_dist=0.3u"nm")
velocities = [random_velocity(atom_mass, temp) for i in 1:n_atoms]
pairwise_inters = (LennardJones(),)
simulator = VelocityVerlet(
  dt=0.002u"ps",
  coupling=AndersenThermostat(temp, 1.0u"ps"),
)

sys = System(
  atoms=atoms,
  coords=coords,
  boundary=boundary,
  velocities=velocities,
  pairwise_inters=pairwise_inters,
  loggers=(
    temp=TemperatureLogger(100),
    coords=CoordinateLogger(10)
  ),
)

simulate!(sys, simulator, 10_000)
