# Connecting the classifier to the course math — Step 8

This explanation uses [the fixed project features/classes](../src/schema.py), [the preprocessing contract](./PREPROCESSING.md), [Assignment 3, Q10–Q11](../Assigments/assignment_3_3.py), and [lecture Sections 8.3–8.5](../documents/COE379L_lecture_notes%20%281%29.pdf). The optional NumPy exercise implements the equations below; it does not change the candidate search or inspect test data.

## 1. Shapes and the model class

After training-fitted preprocessing, each record has 27 numeric features: eight standardized measurements and 19 one-hot indicators. Stack the 1,600 development-training records as rows:

| Quantity | Shape | Meaning |
| --- | --- | --- |
| X | n × d = 1600 × 27 | Transformed predictors; excludes both labels and the ID |
| W | d × K = 27 × 3 | One coefficient vector per class, stored in columns |
| b | K = 3 | Unpenalized class intercepts, broadcast across rows |
| Z | n × K | Linear class scores, or logits |
| P | n × K | Class probabilities; each row sums to one |
| Y | n × K | One-hot true labels, used only in the loss/gradient |

The class order throughout this exercise is healthy, moderate stress, high stress. Row-oriented code transposes the notes' column-oriented sample convention without changing the model:

$$Z = XW + \mathbf{1}b^T, \qquad P_{ik}=\frac{\exp(Z_{ik})}{\sum_{j=1}^K\exp(Z_{ij})}.$$

The predicted class is the index with greatest probability. A pairwise boundary satisfies $x^T(w_j-w_k)+(b_j-b_k)=0$, so it is a hyperplane in transformed feature space. The current model has no learned nonlinear sensor interactions; one-hot encoding represents category offsets, not arbitrary interactions.

## 2. Objective: likelihood plus regularization

The conditional likelihood is the product of the probabilities assigned to the observed labels. Taking its negative logarithm gives cross-entropy; dividing by n makes the data term a mean:

$$J(W,b)=-\frac{1}{n}\sum_{i=1}^n\log P_{i,y_i}+\frac{\lambda}{2}\|W\|_F^2.$$

The intercept is excluded from the penalty. The factor of one-half makes the penalty gradient exactly $\lambda W$. This convention fixes the implementation; a summed-loss objective or a different penalty factor requires a different regularization value to describe the same problem.

For stable computation, subtract the maximum logit in each row before exponentiating:

$$m_i=\max_k Z_{ik},\qquad \log P_{ik}=(Z_{ik}-m_i)-\log\sum_j\exp(Z_{ij}-m_i).$$

Use these log probabilities directly in the loss instead of taking the logarithm of rounded or underflowed probabilities. Very small probabilities may underflow to zero in the returned probability array while the log-probability loss remains finite. Float64 is used for the gradient checks, following [lecture Appendix C.3](../documents/COE379L_lecture_notes%20%281%29.pdf).

## 3. Derive the gradient

For one sample, write its loss as $\ell_i=-\sum_kY_{ik}\log P_{ik}$. The softmax/log-softmax derivatives are:

$$\frac{\partial P_{ik}}{\partial Z_{ij}}=P_{ik}(\delta_{kj}-P_{ij}),\qquad
\frac{\partial\log P_{ik}}{\partial Z_{ij}}=\delta_{kj}-P_{ij}.$$

Because the one-hot row sums to one,

$$\frac{\partial\ell_i}{\partial Z_{ij}}=-\sum_kY_{ik}(\delta_{kj}-P_{ij})=P_{ij}-Y_{ij}.$$

Now apply the chain rule through $Z=XW+\mathbf{1}b^T$. If $G=(P-Y)/n$, then

$$\nabla_W J=X^TG+\lambda W=\frac{X^T(P-Y)}{n}+\lambda W,$$

$$\nabla_b J=G^T\mathbf{1}=\operatorname{mean}_{i}(P_i-Y_i).$$

Thus one full-batch gradient-descent update is

$$W^{t+1}=W^t-\alpha\nabla_WJ,\qquad b^{t+1}=b^t-\alpha\nabla_bJ.$$

[Assignment 3, Q10](../Assigments/assignment_3_3.py) derives the binary **summed** gradient $\sum_i(\sigma(\theta^Tx_i)-y_i)x_i$. Here the scalar residual becomes a vector of class residuals, rows are averaged rather than summed, and the L2 gradient is added. The implementation retains the same residual-times-input structure as least squares; classification changes the output/link and objective.

## 4. Separability, identifiability, and coefficients

On strictly separable data, scaling parameters to increase every correct-class margin drives unregularized cross-entropy toward zero while parameter norms grow without a finite optimum. This is the failure analyzed in [Assignment 3, Q11](../Assigments/assignment_3_3.py). A positive L2 penalty makes increasing weight norm costly. It controls overfitting and stabilizes optimization; it does not create missing information or certify real-world validity. Step 7's observed errors do not establish that these particular sensor inputs are linearly separable.

Softmax also has a representation redundancy: replacing every class weight with $w_k+a$ and every intercept with $b_k+c$ adds the same sample-specific value $x^Ta+c$ to all logits, leaving probabilities unchanged. Lecture Section 8.5.1 fixes a reference class to remove the unregularized model's common-shift freedom.

The **data loss** is invariant to these shifts, but the full L2-penalized objective generally is not invariant to a common weight shift. Symmetric L2 over all K weight columns prefers their minimum-norm centered representative; a shared intercept shift remains unpenalized and can be fixed by centering b. Simply setting a reference-class coefficient to zero and penalizing the remaining columns with the same scalar penalty need not reproduce the symmetric regularized model. Consequently, compare aligned probabilities and matched objective values, not raw coefficients alone. Redundant one-hot indicators and intercepts add further parameter-representation issues.

## 5. Match the installed library's regularization convention

The installed scikit-learn 1.9.1 source was inspected locally. In `sklearn.linear_model._logistic._logistic_regression_path`, L-BFGS uses `l2_reg_strength = 1 / (C * sw_sum)`. `LinearModelLoss` defines the data loss as the weighted mean plus `0.5 * l2_reg_strength * ||W||^2`, excluding intercepts.

For this exercise, sample/class weights are absent, so `sw_sum = n` and the corresponding convention is:

$$\boxed{\lambda=\frac{1}{Cn}}.$$

For C=1 and n=1600, **lambda = 0.000625**. It would be incorrect to substitute lambda=1 merely because C=1. With weights, the denominator depends on the effective sum of weights; this check does not assert one scaling for every solver, library release, or objective convention.

The numerical comparison uses the same development split and training-fitted transformation for both implementations. A reference L-BFGS fit uses the same C/L2 model with a tighter optimizer tolerance solely to compare numerical solutions; it does not replace Step 7's saved candidate or choose model settings using validation. The NumPy optimizer chooses its step size from a training-design smoothness bound, not a validation sweep.

## 6. Numerical verification and outputs

The reproducible experiment and its measured results are recorded below after execution. Central finite differences check every weight/intercept on a tiny seeded float64 example, including the penalty convention. Separate checks cover probability sums, stable loss at extreme logits, common-shift invariance of probabilities, and learning a simple synthetic problem. These checks connect to [Assignment 5, Q2](../Assigments/assignment_5.py); NumPy finite differences verify the analytic derivative without adding JAX.

<!-- NUMERICAL_RESULTS -->

## 7. Measured numerical results

This section is regenerated by the experiment command; the mathematical explanation above is preserved.

| Check | Result |
| --- | ---: |
| Central finite-difference parameters checked | 15 |
| Maximum absolute gradient error | 1.945e-11 |
| Relative L2 gradient error (normalizer at least 1) | 4.320e-11 |
| Finite log-loss for an extreme confidently wrong sample | 2000.0 |
| Accuracy on the simple synthetic training problem | 100.00% |
| NumPy full-batch iterations | 7031 |
| Gradient tolerance reached | True |
| Final full-gradient norm | 9.622e-09 |
| NumPy penalized training objective | 0.788710957930 |
| Reference penalized training objective | 0.788710957930 |
| Absolute objective gap | 4.980e-13 |
| Maximum validation probability difference | 6.164e-07 |
| Validation predicted-label agreement | 100.00% |

The NumPy and L-BFGS solutions agree numerically in probabilities and matched objective value. This verifies the equations and implementation conventions; it does not improve the model class, establish biological validity, or select a final model. Validation was transformed once per fitted preprocessor and never used for optimizer stopping or step-size selection. The tighter reference is a numerical check, and Step 7's C=1 candidate/artifacts remain unchanged.

![Training-only convergence](./figures/softmax_math_convergence.png)

The left panel shows gradient descent lowering the penalized training objective toward the reference value; the right panel shows the full gradient norm reaching the declared tolerance. Both curves use development-train only; their convergence is an optimization result rather than a generalization estimate.

## 8. Reproduce and inspect

```powershell
& ../.venv/Scripts/python.exe -m src.softmax_experiment
& ../.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Run from the repository root. [The NumPy implementation](../src/softmax.py) contains the stable loss/gradient, full-batch optimizer, and finite-difference check. [The experiment](../src/softmax_experiment.py) fits fresh development-training preprocessing, verifies a simple synthetic problem, and compares to a same-objective library reference. It uses one BLAS thread for consistent small-matrix CPU execution, with no new JAX/Optax installation.

[The complete summary](./softmax_math_summary.json) records numeric precision, seeds, source/membership hashes, installed versions, inspected source-function hash, regularization conversion, optimizer settings/history, reference diagnostics, and validation comparisons. [The math notebook](../notebooks/02b_softmax_math.ipynb) presents the derivation and executes the verification. [The local educational artifact](../models/educational_softmax.joblib) saves preprocessing, NumPy parameters, class order, and lambda; it is not a new selected project candidate.

All 29 tests pass, including six dedicated [math tests](../tests/test_softmax.py). Raw data, memberships, Step 7 model artifacts, and the reserved test table remain unchanged. Step 8 is complete, including its optional implementation/verification/experiment. Next is Step 9's declared development-only tuning and selection.
