# Claim boundaries

## Supported by the completed experiment

- The frozen MLP outperformed this frozen GCN-GRU on the clean Net3 held-out benchmark.
- Test leak pipes were disjoint from training leak pipes.
- Predictive entropy was poorly aligned with controlled evidence degradation in this benchmark.
- Attribution-profile divergence was more informative than entropy in most tested degradation conditions, while remaining heterogeneous.
- The same missing-sensor reconstruction improved the MLP and harmed the GCN-GRU.
- The frozen controller reduced unsafe proceeding under the experiment's action semantics.
- BattLeDIM 2018→2019 predictive transfer failed for both frozen models.
- The post-hoc BattLeDIM diagnostic found substantial inter-year covariate shift.

## Not supported

- GNNs are generally worse than MLPs for water networks.
- Attribution divergence is a universal or strong evidence-failure detector.
- Predictive entropy or uncertainty quantification generally fails.
- Sensor reconstruction generally improves reliability.
- The controller makes the external BattLeDIM predictor safe.
- The approach achieves state-of-the-art leak detection.
- The method provides early warning; detection lead time was not the evaluated endpoint.
- Covariate shift caused the 2019 predictive collapse.
- The results establish live utility performance.
