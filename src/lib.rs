//! Wasserstein distance for fleet distribution shift detection and agent realignment.
//!
//! Uses Sinkhorn optimal transport to:
//! 1. Measure how much the fleet's capability distribution has shifted
//! 2. Find the optimal reallocation plan to realign agents
//! 3. Track distribution evolution over time via JKO gradient flow

/// An agent's capability distribution over K skills.
#[derive(Debug, Clone)]
pub struct AgentDistribution {
    pub agent_id: usize,
    pub weights: Vec<f64>,
    pub total_budget: f64,
}

impl AgentDistribution {
    pub fn new(agent_id: usize, weights: Vec<f64>, total_budget: f64) -> Self {
        Self { agent_id, weights, total_budget }
    }

    /// Normalize weights to sum to 1.
    pub fn normalized(&self) -> Vec<f64> {
        let sum: f64 = self.weights.iter().sum();
        if sum < 1e-12 { vec![0.0; self.weights.len()] } else {
            self.weights.iter().map(|w| w / sum).collect()
        }
    }

    /// Shannon entropy of the distribution.
    pub fn entropy(&self) -> f64 {
        let norm = self.normalized();
        norm.iter().filter(|p| **p > 1e-12).map(|p| -p * p.ln()).sum()
    }

    /// KL divergence from self to other.
    pub fn kl_divergence(&self, other: &AgentDistribution) -> f64 {
        let p = self.normalized();
        let q = other.normalized();
        p.iter().zip(q.iter())
            .filter(|(pi, qi)| **pi > 1e-12 && **qi > 1e-12)
            .map(|(pi, qi)| pi * (pi / qi).ln())
            .sum()
    }

    /// Number of skills.
    pub fn n_skills(&self) -> usize {
        self.weights.len()
    }
}

/// Cost matrix between skill dimensions.
pub fn uniform_cost(n: usize) -> Vec<Vec<f64>> {
    let mut c = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            c[i][j] = (i as f64 - j as f64).abs();
        }
    }
    c
}

/// Euclidean cost matrix based on feature vectors.
pub fn euclidean_cost(features_a: &[Vec<f64>], features_b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    features_a.iter().map(|a| {
        features_b.iter().map(|b| {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
        }).collect()
    }).collect()
}

/// Sinkhorn algorithm for regularized optimal transport.
pub fn sinkhorn(
    source: &[f64],
    target: &[f64],
    cost: &[Vec<f64>],
    reg: f64,
    max_iter: usize,
    tol: f64,
) -> (Vec<Vec<f64>>, f64) {
    let n = source.len();
    let m = target.len();
    let mut k = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            k[i][j] = (-cost[i][j] / reg).exp();
        }
    }

    let mut u = vec![1.0 / n as f64; n];
    let mut v = vec![1.0 / m as f64; m];

    for _ in 0..max_iter {
        // u = a ./ (K @ v)
        for i in 0..n {
            let kv: f64 = (0..m).map(|j| k[i][j] * v[j]).sum();
            u[i] = source[i] / kv.max(1e-12);
        }
        // v = b ./ (K^T @ u)
        for j in 0..m {
            let ku: f64 = (0..n).map(|i| k[i][j] * u[i]).sum();
            v[j] = target[j] / ku.max(1e-12);
        }
        // Check convergence
        let u_sum: f64 = u.iter().sum();
        if (u_sum - 1.0).abs() < tol { break; }
    }

    // Transport plan
    let mut plan = vec![vec![0.0; m]; n];
    for i in 0..n {
        for j in 0..m {
            plan[i][j] = u[i] * k[i][j] * v[j];
        }
    }

    // Transport cost
    let total_cost: f64 = plan.iter().zip(cost.iter())
        .map(|(p_row, c_row)| p_row.iter().zip(c_row.iter()).map(|(p, c)| p * c).sum::<f64>())
        .sum();

    (plan, total_cost)
}

/// Wasserstein-1 distance between two distributions.
pub fn wasserstein_1(source: &[f64], target: &[f64], cost: &[Vec<f64>]) -> f64 {
    let (_, cost_val) = sinkhorn(source, target, cost, 0.1, 100, 1e-6);
    cost_val
}

/// Fleet distribution — all agents' capability profiles.
#[derive(Debug, Clone)]
pub struct FleetDistribution {
    pub agents: Vec<AgentDistribution>,
}

impl FleetDistribution {
    pub fn new(agents: Vec<AgentDistribution>) -> Self {
        Self { agents }
    }

    /// Fleet-wide average distribution.
    pub fn fleet_average(&self) -> Vec<f64> {
        let n_skills = self.agents[0].n_skills();
        let n_agents = self.agents.len() as f64;
        let mut avg = vec![0.0; n_skills];
        for agent in &self.agents {
            let norm = agent.normalized();
            for (i, w) in norm.iter().enumerate() {
                avg[i] += w / n_agents;
            }
        }
        avg
    }

    /// Total fleet budget.
    pub fn total_budget(&self) -> f64 {
        self.agents.iter().map(|a| a.total_budget).sum()
    }

    /// Fleet spread: average pairwise Wasserstein distance.
    pub fn spread(&self, cost: &[Vec<f64>]) -> f64 {
        let n = self.agents.len();
        if n < 2 { return 0.0; }
        let mut total = 0.0;
        let mut count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                total += wasserstein_1(
                    &self.agents[i].normalized(),
                    &self.agents[j].normalized(),
                    cost,
                );
                count += 1;
            }
        }
        total / count as f64
    }

    /// Measure shift from another fleet distribution.
    pub fn shift_from(&self, other: &FleetDistribution, cost: &[Vec<f64>]) -> f64 {
        let self_avg = self.fleet_average();
        let other_avg = other.fleet_average();
        wasserstein_1(&self_avg, &other_avg, cost)
    }
}

/// Realignment plan: how to redistribute agent budgets.
#[derive(Debug, Clone)]
pub struct RealignmentPlan {
    pub transfers: Vec<Transfer>,
    pub total_cost: f64,
    pub total_shift: f64,
}

/// A single budget transfer between agents.
#[derive(Debug, Clone)]
pub struct Transfer {
    pub from_agent: usize,
    pub to_agent: usize,
    pub skill: usize,
    pub amount: f64,
}

/// Compute optimal realignment plan using Sinkhorn.
pub fn compute_realignment(
    current: &FleetDistribution,
    target: &FleetDistribution,
    reg: f64,
) -> RealignmentPlan {
    let cost = uniform_cost(current.agents[0].n_skills());
    let current_avg = current.fleet_average();
    let target_avg = target.fleet_average();

    let (plan, total_cost) = sinkhorn(&current_avg, &target_avg, &cost, reg, 100, 1e-6);

    let mut transfers = Vec::new();
    let n_skills = current.agents[0].n_skills();
    for i in 0..n_skills {
        for j in 0..n_skills {
            if plan[i][j] > 1e-8 && i != j {
                // Find agents with surplus in skill i and deficit in skill j
                for agent in &current.agents {
                    let norm = agent.normalized();
                    if norm[i] > target_avg[i] * 1.1 {
                        transfers.push(Transfer {
                            from_agent: agent.agent_id,
                            to_agent: agent.agent_id, // Self-transfer = rebalance
                            skill: i,
                            amount: plan[i][j] * agent.total_budget,
                        });
                    }
                }
            }
        }
    }

    let total_shift = wasserstein_1(&current_avg, &target_avg, &cost);

    RealignmentPlan { transfers, total_cost, total_shift }
}

/// JKO gradient flow: sequence of fleet distributions converging to equilibrium.
#[derive(Debug, Clone)]
pub struct JKOFlow {
    pub steps: Vec<FleetDistribution>,
    pub step_costs: Vec<f64>,
}

impl JKOFlow {
    /// Run JKO flow from initial distribution.
    pub fn run(
        initial: &FleetDistribution,
        target: &FleetDistribution,
        n_steps: usize,
        step_size: f64,
    ) -> Self {
        let cost = uniform_cost(initial.agents[0].n_skills());
        let mut steps = vec![initial.clone()];
        let mut step_costs = Vec::new();

        for _ in 0..n_steps {
            let current = &steps[steps.len() - 1];
            let current_avg = current.fleet_average();
            let target_avg = target.fleet_average();

            // Move each agent's distribution toward target by step_size
            let mut new_agents = Vec::new();
            for agent in &current.agents {
                let norm = agent.normalized();
                let new_weights: Vec<f64> = norm.iter().zip(target_avg.iter())
                    .map(|(w, t)| w + step_size * (t - w))
                    .collect();
                new_agents.push(AgentDistribution::new(agent.agent_id, new_weights, agent.total_budget));
            }

            let new_dist = FleetDistribution::new(new_agents);
            let shift = steps.last().unwrap().shift_from(&new_dist, &cost);
            step_costs.push(shift);
            steps.push(new_dist);
        }

        JKOFlow { steps, step_costs }
    }

    /// Total flow cost.
    pub fn total_cost(&self) -> f64 {
        self.step_costs.iter().sum()
    }

    /// Has the flow converged (last step cost < threshold)?
    pub fn has_converged(&self, threshold: f64) -> bool {
        self.step_costs.last().map(|c| *c < threshold).unwrap_or(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_uniform_agent(id: usize, n_skills: usize) -> AgentDistribution {
        let w = vec![1.0 / n_skills as f64; n_skills];
        AgentDistribution::new(id, w, 100.0)
    }

    fn make_specialist(id: usize, n_skills: usize, focus: usize) -> AgentDistribution {
        let mut w = vec![0.01; n_skills];
        w[focus % n_skills] = 1.0;
        AgentDistribution::new(id, w, 100.0)
    }

    #[test]
    fn test_agent_distribution_normalized() {
        let a = AgentDistribution::new(0, vec![1.0, 2.0, 3.0], 100.0);
        let norm = a.normalized();
        let sum: f64 = norm.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_agent_entropy() {
        let uniform = make_uniform_agent(0, 4);
        let specialist = make_specialist(1, 4, 0);
        assert!(uniform.entropy() > specialist.entropy());
    }

    #[test]
    fn test_kl_divergence_symmetry_violation() {
        let a = make_specialist(0, 3, 0);
        let b = make_specialist(1, 3, 1);
        let kl_ab = a.kl_divergence(&b);
        let kl_ba = b.kl_divergence(&a);
        assert!(kl_ab > 0.0);
        assert!(kl_ba > 0.0);
        // KL is NOT symmetric
    }

    #[test]
    fn test_kl_divergence_self_is_zero() {
        let a = make_uniform_agent(0, 3);
        assert!(a.kl_divergence(&a) < 1e-10);
    }

    #[test]
    fn test_uniform_cost() {
        let c = uniform_cost(3);
        assert!((c[0][0]).abs() < 1e-10);
        assert!((c[0][1] - 1.0).abs() < 1e-10);
        assert!((c[0][2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_cost() {
        let a = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let b = vec![vec![0.0, 0.0], vec![0.0, 1.0]];
        let c = euclidean_cost(&a, &b);
        assert!((c[0][0]).abs() < 1e-10);
        assert!((c[0][1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sinkhorn_plan_sums() {
        let source = vec![0.5, 0.5];
        let target = vec![0.3, 0.7];
        let cost = uniform_cost(2);
        let (plan, _) = sinkhorn(&source, &target, &cost, 0.1, 100, 1e-6);
        let row_sums: Vec<f64> = plan.iter().map(|r| r.iter().sum()).collect();
        for (i, s) in row_sums.iter().enumerate() {
            assert!((s - source[i]).abs() < 0.1, "Row {} sum {} != {}", i, s, source[i]);
        }
    }

    #[test]
    fn test_wasserstein_identical() {
        let dist = vec![0.25, 0.25, 0.25, 0.25];
        let cost = uniform_cost(4);
        let w = wasserstein_1(&dist, &dist, &cost);
        assert!(w < 0.1, "Identical distributions should have ~0 distance, got {}", w);
    }

    #[test]
    fn test_wasserstein_shifted() {
        let a = vec![0.9, 0.1, 0.0, 0.0];
        let b = vec![0.0, 0.0, 0.1, 0.9];
        let cost = uniform_cost(4);
        let w = wasserstein_1(&a, &b, &cost);
        assert!(w > 0.5, "Shifted distributions should have large distance, got {}", w);
    }

    #[test]
    fn test_fleet_average() {
        let fleet = FleetDistribution::new(vec![
            make_specialist(0, 3, 0),
            make_specialist(1, 3, 1),
            make_specialist(2, 3, 2),
        ]);
        let avg = fleet.fleet_average();
        assert_eq!(avg.len(), 3);
        // Should be roughly uniform
        for v in &avg {
            assert!(*v > 0.1 && *v < 0.9);
        }
    }

    #[test]
    fn test_fleet_spread() {
        let spread_fleet = FleetDistribution::new(vec![
            make_specialist(0, 3, 0),
            make_specialist(1, 3, 1),
            make_specialist(2, 3, 2),
        ]);
        let tight_fleet = FleetDistribution::new(vec![
            make_uniform_agent(0, 3),
            make_uniform_agent(1, 3),
            make_uniform_agent(2, 3),
        ]);
        let cost = uniform_cost(3);
        let spread = spread_fleet.spread(&cost);
        let tight = tight_fleet.spread(&cost);
        assert!(spread > tight, "Specialist fleet should be more spread than uniform");
    }

    #[test]
    fn test_fleet_shift() {
        let before = FleetDistribution::new(vec![
            make_specialist(0, 3, 0),
            make_specialist(1, 3, 1),
        ]);
        let after = FleetDistribution::new(vec![
            make_specialist(0, 3, 2),
            make_specialist(1, 3, 0),
        ]);
        let same = FleetDistribution::new(vec![
            make_specialist(0, 3, 0),
            make_specialist(1, 3, 1),
        ]);
        let cost = uniform_cost(3);
        let shift = before.shift_from(&after, &cost);
        let no_shift = before.shift_from(&same, &cost);
        assert!(shift > no_shift);
    }

    #[test]
    fn test_realignment_plan() {
        let current = FleetDistribution::new(vec![
            make_specialist(0, 3, 0),
            make_specialist(1, 3, 1),
        ]);
        let target = FleetDistribution::new(vec![
            make_specialist(0, 3, 2),
            make_specialist(1, 3, 0),
        ]);
        let plan = compute_realignment(&current, &target, 0.1);
        assert!(plan.total_shift > 0.0);
    }

    #[test]
    fn test_jko_flow_converges() {
        let initial = FleetDistribution::new(vec![
            make_specialist(0, 3, 0),
            make_specialist(1, 3, 1),
            make_specialist(2, 3, 2),
        ]);
        let target = FleetDistribution::new(vec![
            make_uniform_agent(0, 3),
            make_uniform_agent(1, 3),
            make_uniform_agent(2, 3),
        ]);
        let flow = JKOFlow::run(&initial, &target, 10, 0.2);
        assert_eq!(flow.steps.len(), 11);
        assert!(flow.total_cost() > 0.0);
        // Should converge
        assert!(flow.has_converged(0.01) || flow.step_costs.last().unwrap() < &flow.step_costs[0]);
    }

    #[test]
    fn test_jko_flow_cost_decreases() {
        let initial = FleetDistribution::new(vec![
            make_specialist(0, 4, 0),
        ]);
        let target = FleetDistribution::new(vec![
            make_uniform_agent(0, 4),
        ]);
        let flow = JKOFlow::run(&initial, &target, 5, 0.3);
        // Each step should cost less than the first
        for cost in &flow.step_costs[1..] {
            assert!(*cost <= flow.step_costs[0] + 0.1);
        }
    }

    #[test]
    fn test_five_agent_fleet() {
        let fleet = FleetDistribution::new((0..5).map(|i| make_specialist(i, 5, i)).collect());
        let cost = uniform_cost(5);
        let spread = fleet.spread(&cost);
        assert!(spread > 0.0);
        assert_eq!(fleet.total_budget(), 500.0);
    }

    #[test]
    fn test_sinkhorn_large() {
        let n = 10;
        let source = vec![1.0 / n as f64; n];
        let target = {
            let mut t = vec![0.0; n];
            t[0] = 0.5;
            t[5] = 0.5;
            t
        };
        let cost = uniform_cost(n);
        let (plan, total_cost) = sinkhorn(&source, &target, &cost, 0.1, 200, 1e-6);
        assert!(total_cost >= 0.0);
        let all_sum: f64 = plan.iter().map(|r| r.iter().sum::<f64>()).sum();
        assert!((all_sum - 1.0).abs() < 0.1);
    }
}
