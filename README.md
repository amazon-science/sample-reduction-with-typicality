# Sample_Reduction_With_Typicality

This package implements the algorithm describes in [this paper](https://www.google.com/search?q=GET+PAPER+URL+HERE). This paper introduces typicality for information retrieval (TIR) for down-sampling training data with high dimensional feature space. High dimensionality and data size are often blockers to fitting complex machine learning models in various domains. TIR allows for an efficient and explainable model training without compromising model performance. We leverage the notion of typicality, which is an empirically derived quantity that resembles probability. We use prototypes, which are the local peaks of the typicality as well as of the data density in a pre-processing step, as a way to extract most information from the training data. We apply the proposed approach for three different models on simulated data as well as real industry data sets. Our results indicate that the downsampled dataset, obtained by our method, retains information content of the overall set and promises high model performance. The following example can be used to mimic the simulation results in the paper.

# Audience

Data Scientists

# Installing sample_reduction_with_typicality
```bash
git clone https://github.com/amazon-science/sample-reduction-with-typicality.git
cd sample_reduction_with_typicality
pip install .
```

# Example

```python
import numpy as np
import pandas as pd
import scipy.stats as sp
from scipy.stats import norm
from sample_reduction_with_typicality import SampleReductionWithTypicality
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

n = 100000
Rho = [
    [1.35071688, 0.15321558, 0.84785951, 0.82255503, -0.33551541, 0.62205449, 0.42880575],
    [0.15321558, 1.35113273, -0.66183342, 0.74442862, 0.67287063, -0.28934146, 0.34474363],
    [0.84785951, -0.66183342, 1.16071755, 0.21553483, -0.54921448, 0.55342434, 0.42030557],
    [0.82255503, 0.74442862, 0.21553483, 1.27186731, 0.80719934, 0.81152044, 0.89989037],
    [-0.33551541, 0.67287063, -0.54921448, 0.80719934, 1.46557044, 0.58029024, 0.74410743],
    [0.62205449, -0.28934146, 0.55342434, 0.81152044, 0.58029024, 1.32526075, 0.60227779],
    [0.42880575, 0.34474363, 0.42030557, 0.89989037, 0.74410743, 0.60227779, 1.09473434],
]
Z = np.random.multivariate_normal([0] * 7, Rho, n)
U = norm.cdf(Z, 0, 1)
X_large_sample = [
    sp.gamma.ppf(U[:, 0], 2, scale=1),
    sp.beta.ppf(U[:, 1], 2, 2),
    sp.t.ppf(U[:, 2], 5),
    sp.gamma.ppf(U[:, 3], 2, scale=1),
    sp.t.ppf(U[:, 4], 2),
    sp.t.ppf(U[:, 5], 3),
    sp.t.ppf(U[:, 6], 15),
]
beta = [0.1, 2, -0.5, 2, 0.3, 0.4, 10]
sigma = np.random.normal(0, 10, len(X_large_sample[0]))
y = (
    beta[0] * X_large_sample[0]
    + beta[1] * X_large_sample[1]
    + beta[2] * X_large_sample[2]
    + beta[3] * X_large_sample[3] ** 2
    + beta[4] * X_large_sample[4]
    + beta[5] * X_large_sample[5]
    + beta[6] * X_large_sample[6]
    + sigma
)
X_large = [
    X_large_sample[0],
    X_large_sample[1],
    X_large_sample[2],
    X_large_sample[3],
    X_large_sample[4],
    X_large_sample[5],
    X_large_sample[6],
    y,
]
data = pd.DataFrame(
    {
        "x0": X_large[0],
        "x1": X_large[1],
        "x2": X_large[2],
        "x3": X_large[3],
        "x4": X_large[4],
        "x5": X_large[5],
        "x6": X_large[6],
        "y": X_large[7],
    }
)
# Reduce
srwt = SampleReductionWithTypicality(batch_size=20000, verbose=True)
X_final_large = srwt.reduce(data.to_numpy())
log.info(f"Start number of rows: {n}. End number of rows: {X_final_large.shape}")
log.info("Testing SampleReductionWithTypicality class")
```

# License Summary

This sample code is made available under a MIT-0 license. See the [LICENSE](https://github.com/aws/sample_reduction_with_typicality/blob/master/LICENSE) file.
