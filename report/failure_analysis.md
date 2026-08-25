# Failure analysis

The experiment produced several results that I did not try to tune away.

## The GCN-GRU was weaker than the MLP on clean Net3

The graph model did not outperform the simpler MLP on the clean held-out benchmark. The result is specific to these frozen architectures, features, training rules, and split. It does not support a general statement that graph models are worse for water-network prediction.

## Predictive entropy did not track evidence failure well

The median degradation-detection AUROC for entropy was approximately chance in both the controlled Net3 study and the BattLeDIM external study.

This only evaluates the implemented binary predictive-entropy signal. It does not establish that uncertainty quantification in general is ineffective.

## Attribution divergence was useful in some conditions and weak in others

The attribution-profile signal beat entropy in most matched degradation conditions, but its median AUROC remained modest. High missingness and some bias conditions were highly separable; several noise and stale conditions were not.

I therefore treat it as a complementary signal, not as a general detector.

## Reconstruction helped one architecture and harmed the other

The training-only Ridge reconstruction substantially improved the MLP under missing pressure sensors. For the GCN-GRU, the same intervention became increasingly harmful as more sensors were missing.

The experiment does not identify the causal mechanism behind that difference. One possible explanation is the interaction between imputed values and the graph model's explicit observation mask, but that would require a separate experiment.

## External predictive transfer failed

The BattLeDIM 2018-trained models were approximately chance in the held-out 2019 year.

A post-hoc diagnostic shows strong inter-year feature shift across most SCADA variables, but that is an association. I do not claim that the measured covariate shift alone caused the predictive collapse.

## Reliability monitoring cannot replace predictive validity

The external controller can still defer some incorrect predictions, and attribution divergence can still react to degraded inputs. That does not rescue an end-to-end system whose clean predictor no longer discriminates in the target domain.

This is the main boundary exposed by the external study.
