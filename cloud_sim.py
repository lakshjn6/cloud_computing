"""
Phase 3 — Cloud Data Center Simulation (SimPy)
================================================
Models a cloud data centre with N physical hosts.

Two autoscaling controllers are compared:

  PREDICTIVE (AI-driven)
    Uses the ML model's next-step forecast to provision capacity
    ONE STEP AHEAD of demand — exactly like a real proactive system.
    If the model predicts CPU will be 80% next interval, we size
    the host pool *now* to handle 80% (+ safety buffer).

  REACTIVE (baseline / traditional)
    Scales only on the current observed demand — no look-ahead.
    This is how most naive threshold-based autoscalers work today.

The predictive controller should show fewer SLA violations because
it reacts before the spike, while reactive scrambles after.
"""

import simpy
import joblib
import numpy as np
import pandas as pd
import os


# -----------------------------------------------------------------------
class PhysicalHost:
    """One physical server rack in the data centre."""
    HOST_CPU_CAPACITY_PCT = 10.0   # each host can absorb 10 % of cluster demand

    def __init__(self, host_id, total_cpu_cores=32, total_ram_gb=128):
        self.host_id             = host_id
        self.total_cpu_cores     = total_cpu_cores
        self.total_ram_gb        = total_ram_gb
        self.allocated_cpu_cores = 0.0
        self.allocated_ram_gb    = 0.0

    @property
    def cpu_utilization(self):
        return (self.allocated_cpu_cores / self.total_cpu_cores) * 100.0

    @property
    def ram_utilization(self):
        return (self.allocated_ram_gb / self.total_ram_gb) * 100.0


# -----------------------------------------------------------------------
class CloudDataCenterSim:
    """
    SimPy discrete-event simulation of a cloud data centre.

    Capacity model
    --------------
    Each physical host can absorb HOST_CPU_CAPACITY_PCT (10 %) of total
    cluster demand.  So if the cluster receives 80 % CPU demand we need
    at least ceil(80 / 10) = 8 hosts to avoid SLA breach.

    The predictive controller sizes the pool using the ML prediction for
    the *next* time step (look-ahead by 1 interval = 5 minutes).
    The reactive controller sizes using the *current* actual demand.
    """

    CAPACITY_PER_HOST = 10.0     # % demand each host handles
    SAFETY_BUFFER     = 1.20     # 20 % safety margin on top of forecast

    def __init__(
        self,
        env,
        workload_series,
        model_predictions=None,
        initial_hosts=8,
        mode="predictive",
    ):
        self.env               = env
        self.workload_series   = np.array(workload_series, dtype=float)
        # Shift predictions by +1 so they represent the NEXT step's demand
        # (this is the true proactive look-ahead)
        if model_predictions is not None:
            mp = np.array(model_predictions, dtype=float)
            # lookahead: prediction[i] is used at step i to prepare for step i+1
            self.model_predictions = np.roll(mp, -1)
            self.model_predictions[-1] = mp[-1]   # repeat last pred at boundary
        else:
            self.model_predictions = None
        self.mode = mode

        self.active_hosts = [
            PhysicalHost(i) for i in range(initial_hosts)
        ]

        # Telemetry
        self.logs                        = []
        self.sla_violations              = 0
        self.over_provisioned_cpu_total  = 0.0
        self.under_provisioned_cpu_total = 0.0
        self.proactive_true_positives    = 0   # scaled up before an actual spike

    # ------------------------------------------------------------------
    def _planning_demand(self, step_idx):
        """Return the demand value the controller acts upon."""
        if (
            self.mode == "predictive"
            and self.model_predictions is not None
            and step_idx < len(self.model_predictions)
        ):
            return float(self.model_predictions[step_idx])
        return float(self.workload_series[step_idx])

    # ------------------------------------------------------------------
    def run_simulation_step(self, step_idx):
        actual_demand   = float(self.workload_series[step_idx])
        planning_demand = self._planning_demand(step_idx)

        # ---- Autoscaling decision (based on planning_demand) ----------
        required_hosts = int(
            np.ceil((planning_demand * self.SAFETY_BUFFER) / self.CAPACITY_PER_HOST)
        )
        required_hosts = max(2, min(required_hosts, 80))

        current_count  = len(self.active_hosts)
        scaling_action = "HOLD"

        if required_hosts > current_count:
            delta = required_hosts - current_count
            for _ in range(delta):
                self.active_hosts.append(PhysicalHost(len(self.active_hosts)))
            scaling_action = f"SCALE_UP +{delta}"
            # True positive: we scaled up AND actual next demand is a spike (>70%)
            if self.mode == "predictive" and actual_demand > 70.0:
                self.proactive_true_positives += 1

        elif required_hosts < current_count:
            removable = min(current_count - required_hosts, current_count - 2)
            for _ in range(removable):
                self.active_hosts.pop()
            scaling_action = f"SCALE_DOWN -{removable}"

        # ---- SLA audit (based on actual_demand vs actual capacity) ----
        total_capacity = len(self.active_hosts) * self.CAPACITY_PER_HOST

        # VM count estimate: each host runs up to 5 VMs
        vm_count = len(self.active_hosts) * 5

        sla_violated = actual_demand > total_capacity
        if sla_violated:
            self.sla_violations += 1
            deficit = actual_demand - total_capacity
            self.under_provisioned_cpu_total += deficit
        else:
            self.over_provisioned_cpu_total += (total_capacity - actual_demand)

        utilisation = min(100.0, (actual_demand / total_capacity) * 100.0)

        self.logs.append({
            "step"                   : step_idx,
            "sim_time_min"           : step_idx * 5,
            "actual_demand_pct"      : round(actual_demand, 3),
            "predicted_demand_pct"   : round(planning_demand, 3),
            "active_hosts"           : len(self.active_hosts),
            "active_vms"             : vm_count,
            "cluster_capacity_pct"   : round(total_capacity, 3),
            "cluster_utilization_pct": round(utilisation, 2),
            "sla_violation"          : sla_violated,
            "scaling_action"         : scaling_action,
            "cpu_over_prov"          : round(max(0, total_capacity - actual_demand), 3),
            "cpu_under_prov"         : round(max(0, actual_demand - total_capacity), 3),
        })


# -----------------------------------------------------------------------
def run_cloud_simulation(mode="predictive", artifacts_path="model_artifacts.joblib"):
    """Run one full simulation pass.  Returns (sim, logs_df)."""
    print(f"\n[SIM] Mode: {mode.upper()}")

    if os.path.exists(artifacts_path):
        art        = joblib.load(artifacts_path)
        y_actual   = art["y_actual"]
        key_map    = {
            "ARIMA Baseline"  : "ARIMA",
            "XGBoost Regressor": "XGBoost",
            "LSTM Neural Net" : "LSTM",
        }
        best_key   = key_map.get(art["best_model_name"], "XGBoost")
        model_preds= art["predictions"].get(best_key, list(art["predictions"].values())[0])
    else:
        print("[WARN] Artifacts not found — using random demo data.")
        np.random.seed(0)
        y_actual    = np.random.uniform(20, 90, 400)
        model_preds = y_actual + np.random.normal(0, 4, 400)

    env = simpy.Environment()
    sim = CloudDataCenterSim(
        env=env,
        workload_series=y_actual,
        model_predictions=model_preds,
        mode=mode,
    )
    for step in range(len(y_actual)):
        sim.run_simulation_step(step)

    logs_df  = pd.DataFrame(sim.logs)
    total    = len(y_actual)
    sla_rate = (sim.sla_violations / total) * 100.0

    print(f"  Steps           : {total}  ({(total*5)/60:.1f} h simulated)")
    print(f"  Avg utilisation : {logs_df['cluster_utilization_pct'].mean():.2f}%")
    print(f"  SLA violations  : {sim.sla_violations} ({sla_rate:.2f}%)")
    print(f"  Over-prov CPU   : {sim.over_provisioned_cpu_total:.1f} units")
    print(f"  True-pos proact : {sim.proactive_true_positives}")
    return sim, logs_df


# -----------------------------------------------------------------------
def compare_predictive_vs_reactive():
    """Run both modes and save the comparison bundle."""
    sim_pred, df_pred = run_cloud_simulation(mode="predictive")
    sim_reac, df_reac = run_cloud_simulation(mode="reactive")

    summary = {
        "Predictive_SLA_Violations" : sim_pred.sla_violations,
        "Reactive_SLA_Violations"   : sim_reac.sla_violations,
        "Predictive_Wasted_CPU"     : sim_pred.over_provisioned_cpu_total,
        "Reactive_Wasted_CPU"       : sim_reac.over_provisioned_cpu_total,
        "Predictive_Avg_Util"       : df_pred["cluster_utilization_pct"].mean(),
        "Reactive_Avg_Util"         : df_reac["cluster_utilization_pct"].mean(),
        "Predictive_TruePos"        : sim_pred.proactive_true_positives,
        "df_predictive"             : df_pred,
        "df_reactive"               : df_reac,
    }
    joblib.dump(summary, "simulation_results.joblib")

    print("\n" + "=" * 60)
    print("  PREDICTIVE vs REACTIVE — SUMMARY")
    print("=" * 60)
    print(f"  {'Metric':<35} {'Predictive':>10}  {'Reactive':>10}")
    print(f"  {'-'*57}")
    print(f"  {'SLA Violations':<35} {sim_pred.sla_violations:>10}  {sim_reac.sla_violations:>10}")
    print(f"  {'Over-provisioned CPU (units)':<35} {sim_pred.over_provisioned_cpu_total:>10.1f}  {sim_reac.over_provisioned_cpu_total:>10.1f}")
    print(f"  {'Avg Cluster Utilisation (%)':<35} {df_pred['cluster_utilization_pct'].mean():>10.2f}  {df_reac['cluster_utilization_pct'].mean():>10.2f}")
    print(f"  {'True-Positive Proactive Scales':<35} {sim_pred.proactive_true_positives:>10}  {'N/A':>10}")
    print("=" * 60)
    print("[SIM] Results saved to 'simulation_results.joblib'.")
    return summary


if __name__ == "__main__":
    compare_predictive_vs_reactive()
