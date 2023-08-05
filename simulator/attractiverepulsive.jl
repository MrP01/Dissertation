struct AttractiveRepulsive{S,C,W,WS,F,E} <: PairwiseInteraction
  cutoff::C
  use_neighbors::Bool
  lorentz_mixing::Bool
  weight_special::W
  weight_solute_solvent::WS
  force_units::F
  energy_units::E
end

function AttractiveRepulsive(;
  cutoff=NoCutoff(),
  use_neighbors=false,
  lorentz_mixing=true,
  weight_special=1,
  weight_solute_solvent=1,
  force_units=u"kJ * mol^-1 * nm^-1",
  energy_units=u"kJ * mol^-1",
  skip_shortcut=false)
  return AttractiveRepulsive{skip_shortcut,typeof(cutoff),typeof(weight_special),
    typeof(weight_solute_solvent),typeof(force_units),typeof(energy_units)}(
    cutoff, use_neighbors, lorentz_mixing, weight_special, weight_solute_solvent,
    force_units, energy_units)
end

use_neighbors(inter::AttractiveRepulsive) = inter.use_neighbors

is_solute(at::Atom) = at.solute
is_solute(at) = false

function Base.zero(lj::AttractiveRepulsive{S,C,W,WS,F,E}) where {S,C,W,WS,F,E}
  return AttractiveRepulsive{S,C,W,WS,F,E}(
    lj.cutoff,
    false,
    false,
    zero(W),
    zero(WS),
    lj.force_units,
    lj.energy_units,
  )
end

function Base.:+(l1::AttractiveRepulsive{S,C,W,WS,F,E},
  l2::AttractiveRepulsive{S,C,W,WS,F,E}) where {S,C,W,WS,F,E}
  return AttractiveRepulsive{S,C,W,WS,F,E}(
    l1.cutoff,
    l1.use_neighbors,
    l1.lorentz_mixing,
    l1.weight_special + l2.weight_special,
    l1.weight_solute_solvent + l2.weight_solute_solvent,
    l1.force_units,
    l1.energy_units,
  )
end

@inline @inbounds function force(inter::AttractiveRepulsive{S,C},
  dr,
  coord_i,
  coord_j,
  atom_i,
  atom_j,
  boundary,
  special::Bool=false) where {S,C}
  if !S && (iszero_value(atom_i.ϵ) || iszero_value(atom_j.ϵ) ||
            iszero_value(atom_i.σ) || iszero_value(atom_j.σ))
    return ustrip.(zero(coord_i)) * inter.force_units
  end

  # Lorentz-Berthelot mixing rules use the arithmetic average for σ
  # Otherwise use the geometric average
  σ = inter.lorentz_mixing ? (atom_i.σ + atom_j.σ) / 2 : sqrt(atom_i.σ * atom_j.σ)
  if (is_solute(atom_i) && !is_solute(atom_j)) || (is_solute(atom_j) && !is_solute(atom_i))
    ϵ = inter.weight_solute_solvent * sqrt(atom_i.ϵ * atom_j.ϵ)
  else
    ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)
  end

  cutoff = inter.cutoff
  r2 = sum(abs2, dr)
  σ2 = σ^2
  params = (σ2, ϵ)

  f = force_divr_with_cutoff(inter, r2, params, cutoff, coord_i, inter.force_units)
  if special
    return f * dr * inter.weight_special
  else
    return f * dr
  end
end

function force_divr(::AttractiveRepulsive, r2, invr2, (σ2, ϵ))
  six_term = (σ2 * invr2)^3
  return (24ϵ * invr2) * (2 * six_term^2 - six_term)
end

@inline @inbounds function potential_energy(inter::AttractiveRepulsive{S,C},
  dr,
  coord_i,
  coord_j,
  atom_i,
  atom_j,
  boundary,
  special::Bool=false) where {S,C}
  if !S && (iszero_value(atom_i.ϵ) || iszero_value(atom_j.ϵ) ||
            iszero_value(atom_i.σ) || iszero_value(atom_j.σ))
    return ustrip(zero(coord_i[1])) * inter.energy_units
  end

  σ = inter.lorentz_mixing ? (atom_i.σ + atom_j.σ) / 2 : sqrt(atom_i.σ * atom_j.σ)
  if (is_solute(atom_i) && !is_solute(atom_j)) || (is_solute(atom_j) && !is_solute(atom_i))
    ϵ = inter.weight_solute_solvent * sqrt(atom_i.ϵ * atom_j.ϵ)
  else
    ϵ = sqrt(atom_i.ϵ * atom_j.ϵ)
  end

  cutoff = inter.cutoff
  r2 = sum(abs2, dr)
  σ2 = σ^2
  params = (σ2, ϵ)

  pe = potential_with_cutoff(inter, r2, params, cutoff, coord_i, inter.energy_units)
  if special
    return pe * inter.weight_special
  else
    return pe
  end
end

function potential(::AttractiveRepulsive, r2, invr2, (σ2, ϵ))
  six_term = (σ2 * invr2)^3
  return 4ϵ * (six_term^2 - six_term)
end
