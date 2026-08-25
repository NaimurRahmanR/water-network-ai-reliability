# Limitations

- The controlled benchmark is a WNTR/EPANET simulation, not a live utility deployment.
- The primary predictive task is binary leak detection rather than leak localization, hydraulic state estimation, or operational control.
- Only one MLP and one GCN-GRU architecture are evaluated.
- The graph model was deliberately not retuned after the held-out comparison.
- Many windows overlap in time. Statistical comparison of the clean models therefore clusters by physical context rather than treating windows as independent.
- Degradation-signal AUROCs and controller rates are descriptive held-out results; they do not have post-hoc significance claims.
- Input×Gradient is one operational definition of attribution instability, not a complete treatment of explanation reliability.
- The missing-sensor reconstruction is a simple linear Ridge method.
- `VERIFY`, `REQUEST_SENSOR`, and `ESCALATE` are deferral actions rather than implemented downstream operational workflows.
- The external BattLeDIM predictor fails in 2019, so the external study does not demonstrate useful end-to-end leak-onset detection.
- The 2018→2019 drift analysis is post-hoc and descriptive. It does not establish causation.
- The study does not support state-of-the-art, field-validity, or general-GNN claims.
