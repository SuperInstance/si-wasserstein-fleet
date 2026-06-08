# si-wasserstein-fleet

> **Proof of Concept:** Wasserstein distance measures fleet distribution shift — Sinkhorn optimal transport finds the cheapest way to realign agents.

## The Insight

A fleet of agents has a **distribution** over capabilities. When agents drift, the fleet distribution shifts. The Wasserstein distance (Earth Mover's Distance) tells you:

1. **How much** the fleet has shifted
2. **Which direction** it shifted
3. **What it costs** to realign

The Sinkhorn algorithm solves this efficiently via entropy-regularized optimal transport.

## What This Proves

- **Fleet spread** = average pairwise Wasserstein distance between agents
- **Distribution shift** = Wasserstein distance from fleet average to target
- **Realignment cost** = minimum transport cost to redistribute capabilities
- **JKO gradient flow** = the natural path the fleet takes to equilibrium

## Usage

```rust
use si_wasserstein_fleet::*;

// Create fleet with specialist agents
let fleet = FleetDistribution::new(vec![
    AgentDistribution::new(0, vec![0.9, 0.1, 0.0], 100.0), // Agent 0: skill 0 specialist
    AgentDistribution::new(1, vec![0.0, 0.5, 0.5], 100.0), // Agent 1: balanced 1,2
    AgentDistribution::new(2, vec![0.1, 0.1, 0.8], 100.0), // Agent 2: skill 2 specialist
]);

let cost = uniform_cost(3);
let spread = fleet.spread(&cost);
println!("Fleet spread: {:.3}", spread);

// Compute Wasserstein distance to target distribution
let target = FleetDistribution::new(vec![
    AgentDistribution::new(0, vec![0.33, 0.33, 0.34], 100.0),
    AgentDistribution::new(1, vec![0.33, 0.33, 0.34], 100.0),
    AgentDistribution::new(2, vec![0.33, 0.33, 0.34], 100.0),
]);
let shift = fleet.shift_from(&target, &cost);
println!("Distribution shift: {:.3}", shift);

// JKO flow: natural convergence path
let flow = JKOFlow::run(&fleet, &target, 10, 0.2);
println!("Converged: {}", flow.has_converged(0.01));
```

## Modules

- `AgentDistribution` — per-agent capability profile with entropy and KL divergence
- `FleetDistribution` — fleet-wide distribution with spread and shift metrics
- `sinkhorn()` — entropy-regularized optimal transport solver
- `wasserstein_1()` — Wasserstein-1 distance
- `compute_realignment()` — optimal budget redistribution plan
- `JKOFlow` — Jordan-Kinderlehrer-Otto gradient flow for distribution evolution

## Connection to Conservation Law

The Wasserstein framework unifies with the conservation law:
- **Transport cost** = γ budget (expensive, long-range realignment)
- **Step size** = η budget (cheap, local adjustments)
- **JKO flow** = the conservation-optimal path to fleet equilibrium

## Tests: 17

Covers: normalization, entropy, KL divergence, cost matrices, Sinkhorn convergence, Wasserstein distances, fleet spread/shift, realignment plans, JKO flow convergence.

## License

MIT
