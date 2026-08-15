# Results Summary

The figures below come from the final holdout evaluation in the original academic project. They represent simulated business value under the project assumptions, not realized company profit.

## House-price regression

The selected pricing decision rule combined the Random Forest regression estimate with the existing baseline offer rule.

- Baseline holdout profit: USD 298,795,171
- Selected decision-rule profit: USD 652,022,456
- Improvement over baseline: approximately 118.2%

## Insurance classification

The selected model was a TPOT-SMOTENC Random Forest Classifier. The probability threshold was selected using the valuation sample rather than the final holdout sample.

- Selected probability threshold: approximately 0.1153
- Baseline holdout profit: USD 5,181,000
- Selected model holdout profit: USD 22,733,448
- Improvement over baseline: approximately 338.8%

These results show why model selection based only on predictive accuracy may be insufficient. The final recommendations were based on the financial consequences of each decision.
