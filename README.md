# ROM-benchmarking
Here I summarize all the order reduction methods trained during the research process of the 2021-2022 course, and I compare them in terms of reconstruction capabilities and orthogonality and interpretability of the modes.


* POD,
* DMD,
* basic AE,
* CNN-based AE,
* CNN-based HAE,
* β-VAE, where it is also relevant to check the parametric study on beta, and the effect of a L1-regularization of the latent space, and
* HVAE,
* Symmetry AE.

**Input data:** The dataset is obtained from a DNS simulation of 2D viscous flow past two colinear plates, aligned perpendicular to the freestream velocity. The plates each have unit length, and the gap between them is also unity. The Reynolds number (based on freestream velocity and the length of one plate) is 100. Due to computational limitations, a low-resolution version of the data is used.

**References:**

[1] H. Eivazi (et al). Towards extraction of orthogonal and parsimonious non-linear modes from turbulent flows (2021)

[2] S. Dawson (et al). Reference for the input data.
