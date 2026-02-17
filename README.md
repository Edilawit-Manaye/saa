# Complete Study Guide: Neural Network Project (Scratch to Advanced)

**Use this to prepare for your presentation. The instructor may ask deep questions—this guide covers everything from basic terms to advanced concepts.**

---

# PART 1: TERMINOLOGY (Start Here)

## 1.1 Basic Terms

| Term | Simple Definition | In Our Project |
|------|-------------------|----------------|
| **Neural Network** | A mathematical model that learns from data by adjusting numbers (weights). It has layers of "neurons" that compute outputs from inputs. | Our network has 5 layers: input → 3 hidden → output. |
| **Layer** | A set of neurons that receive input, do one computation, and produce output. | We have: Input (64), Hidden1 (128), Hidden2 (64), Hidden3 (32), Output (10). |
| **Neuron** | One unit in a layer. It takes many inputs, multiplies by weights, adds a bias, then applies an activation. | Each number in A1, A2, A3 is one "neuron output." |
| **Weight (W)** | A number that multiplies an input. Learning = adjusting these. | W1, W2, W3, W4 are matrices of weights. |
| **Bias (b)** | A number added after the weighted sum. Shifts the activation. | b1, b2, b3, b4 are vectors. |
| **Activation function** | A function applied to the weighted sum (e.g., ReLU, softmax). Adds non-linearity. | We use ReLU in hidden layers, softmax in output. |
| **Forward propagation** | Passing data from input layer to output layer, layer by layer. | `forwardprop.py`: X → Z1→A1 → Z2→A2 → Z3→A3 → Z4→A4. |
| **Backward propagation (backprop)** | Computing how much each weight contributed to the error, using the chain rule. | `backprop.py`: gradients dW1…dW4, db1…db4. |
| **Gradient** | The direction and rate of change of the loss with respect to a parameter. "How to change W to reduce loss." | dW1, db1, …, dW4, db4. |
| **Loss (cost)** | A single number measuring how wrong the predictions are. We minimize it. | Cross-entropy + L2 term. |
| **Optimizer** | The rule that updates weights using gradients (e.g., subtract gradient × learning rate). | We have: GD, RMSProp, Adam, Newton, Muon. |
| **Learning rate** | Step size for weight updates. Too large = unstable; too small = slow. | e.g., 0.1 for GD, 0.001 for Adam. |
| **Epoch** | One full pass over the entire training dataset. | We use 150 epochs. |
| **Batch / Mini-batch** | A subset of data used for one gradient computation and update. | We use batch_size=64. |
| **Classification** | Predicting a category (e.g., digit 0–9). | Digits dataset: 10 classes. |
| **Regression** | Predicting a continuous number. | We did not use regression. |

## 1.2 Our-Specific Terms

| Term | Meaning |
|------|--------|
| **Digits dataset** | 8×8 images of handwritten digits (0–9). 64 features per sample, 10 classes. From sklearn. |
| **One-hot encoding** | Label 3 → [0,0,0,1,0,0,0,0,0,0]. Used for cross-entropy. |
| **L2 regularization** | Penalty = λ × sum of (all weights²). Prevents weights from growing too large; reduces overfitting. |
| **Dropout** | During training, randomly set some neuron outputs to 0 (e.g., 20%). Reduces overfitting. |
| **He initialization** | Weights initialized with scale √(2/n_in). Good for ReLU. |
| **Cross-entropy loss** | For classification: −Σ y_true × log(y_pred). Measures how wrong the predicted probabilities are. |

---

# PART 2: WHAT THE PROJECT DOES (Big Picture)

1. **Data**: Load Digits (1797 samples, 64 features, 10 classes). Split train/test. Scale features.
2. **Model**: One neural network: 64 → 128 → 64 → 32 → 10 (3 hidden layers).
3. **Training**: For each mini-batch:
   - **Forward**: Compute predictions (forwardprop).
   - **Backward**: Compute gradients (backprop).
   - **Update**: Optimizer updates W and b using gradients.
4. **Repeat** for many epochs until loss is low and accuracy is high (~97–98%).
5. **Compare**: We train the same network with 4 optimizers (GD, RMSProp, Adam, Newton) and plot learning curves. Bonus: Muon vs Adam.

---

# PART 3: FORWARD PROPAGATION (Deep Dive)

## 3.1 What Is Forward Propagation?

We pass input **X** through the network layer by layer until we get **A4** (predicted probabilities for each class).

## 3.2 Layer-by-Layer (With Shapes in Our Code)

- **Input**: X has shape `(batch_size, 64)` = (m, 64).

- **Layer 1 (Input → Hidden1)**  
  - Z1 = X @ W1 + b1  
  - Shapes: (m, 64) @ (64, 128) + (1, 128) → (m, 128)  
  - A1 = ReLU(Z1)  
  - Then dropout (if training): randomly zero some entries and scale by 1/(1−p).

- **Layer 2 (Hidden1 → Hidden2)**  
  - Z2 = A1 @ W2 + b2  → (m, 128) @ (128, 64) → (m, 64)  
  - A2 = ReLU(Z2), then dropout.

- **Layer 3 (Hidden2 → Hidden3)**  
  - Z3 = A2 @ W3 + b3  → (m, 64) @ (64, 32) → (m, 32)  
  - A3 = ReLU(Z3), then dropout.

- **Layer 4 (Hidden3 → Output)**  
  - Z4 = A3 @ W4 + b4  → (m, 32) @ (32, 10) → (m, 10)  
  - A4 = softmax(Z4) → (m, 10) probabilities per sample.

## 3.3 Why ReLU?

- **ReLU(x) = max(0, x)**  
- Pros: simple, fast, avoids vanishing gradient for x>0, many zeros → sparsity.  
- Derivative: 1 if x>0, 0 if x≤0. So in backprop we do `dZ = dA * (Z > 0)`.

## 3.4 Why Softmax on Output?

- Converts Z4 (logits) to **probabilities** that sum to 1 per sample.  
- Formula: softmax(z_i) = exp(z_i) / Σ exp(z_j).  
- We use `exp(z - max(z))` for numerical stability.  
- Good for multi-class classification with cross-entropy.

## 3.5 Why Dropout?

- **During training**: For each hidden layer we multiply activations by a random mask (0 or 1/(1−p)). So some neurons are "dropped" (output 0).  
- **During test**: No dropout; we use all neurons.  
- **Effect**: Reduces overfitting by preventing co-adaptation of neurons; acts like training many smaller networks and averaging.

**Instructor might ask**: "What happens to the gradient at dropped neurons?"  
**Answer**: In backward pass we multiply by the **same** dropout mask. So gradients for dropped neurons are 0; they don’t get updated that step.

---

# PART 4: BACKWARD PROPAGATION (Deep Dive)

## 4.1 What Is Backward Propagation?

We have a **loss L**. We want ∂L/∂W and ∂L/∂b for every layer so we can update parameters. Backprop computes these using the **chain rule**: layer by layer from output back to input.

## 4.2 Chain Rule Idea

If L depends on A4, A4 depends on Z4, Z4 depends on W4, then:

- ∂L/∂W4 = (∂L/∂A4) × (∂A4/∂Z4) × (∂Z4/∂W4)

We compute "local" derivatives at each layer and multiply them backward.

## 4.3 Output Layer (Layer 4)

- **Loss**: Cross-entropy with softmax has a very simple derivative: **dZ4 = A4 − y** (then we divide by m for average).  
- So: dW4 = A3.T @ dZ4 + (λ/m)*W4  (the extra term is from L2).  
- db4 = sum(dZ4, axis=0).

**Instructor might ask**: "Why is the gradient A4 − y for softmax + cross-entropy?"  
**Answer**: It comes from the derivative of cross-entropy with respect to the logits; the softmax derivative simplifies so that the combined gradient is (prediction − target).

## 4.4 Hidden Layers (3, 2, 1)

- From layer ℓ+1 we have dZ_{ℓ+1}. We need dZ_ℓ.  
- dA_ℓ = dZ_{ℓ+1} @ W_{ℓ+1}.T  
- Then back through dropout: dA_ℓ = dA_ℓ * dropout_mask_ℓ (same mask as forward).  
- Then back through ReLU: dZ_ℓ = dA_ℓ * (Z_ℓ > 0).  
- Then: dW_ℓ = A_{ℓ−1}.T @ dZ_ℓ + (λ/m)*W_ℓ,  db_ℓ = sum(dZ_ℓ, axis=0).

So we go: output → layer 3 → layer 2 → layer 1, each time using the previous layer’s gradient and the chain rule.

## 4.5 L2 Regularization in Backprop

- Loss has an extra term: (λ/2m) Σ W².  
- Its derivative w.r.t. W is (λ/m) W.  
- So we **add (λ/m)*W** to the gradient of the cross-entropy part.  
- Effect: weights are "pulled" toward zero; prevents them from growing too large.

**Instructor might ask**: "Where do you add L2 in backprop?"  
**Answer**: In the gradient of the weights: e.g. `dW4 = A3.T @ dZ4 + (lambda_reg / m) * W4`. Same idea for W1, W2, W3.

---

# PART 5: LOSS FUNCTION

## 5.1 Cross-Entropy

- For one sample with true class c: **−log(p_c)** where p_c is predicted probability for class c.  
- For many samples: average over the batch.  
- Lower when the model assigns high probability to the correct class.

## 5.2 L2 Regularization Term

- **reg_term = (λ / 2m) × (sum of all W1² + W2² + W3² + W4²)**  
- Total loss = cross_entropy_loss + reg_term.  
- We use λ = 0.01 in the project.

---

# PART 6: OPTIMIZERS (Simple → Advanced)

## 6.1 Gradient Descent (GD)

- **Update**: W_new = W_old − learning_rate × gradient  
- **Idea**: Take a step in the direction that reduces the loss.  
- **Pros**: Simple, stable.  
- **Cons**: Same learning rate for every parameter; can be slow or get stuck in plateaus.

**In code**: `weights[i] -= self.lr * grads_W[i]`

## 6.2 RMSProp

- **Idea**: Keep a running average of **squared** gradients. Divide the gradient by the square root of this average before updating.  
- **Formula**: s = decay×s + (1−decay)×g²;  W -= lr × g / (√s + ε)  
- **Effect**: Parameters with big gradients get smaller effective steps; parameters with small gradients get larger steps. Adapts to the "scale" of each parameter.  
- **decay** (e.g. 0.9): how much we trust the old average vs the new gradient.

**In code**: We maintain `sq_W`, `sq_b`; update with squared gradients; then divide gradient by √(sq) when updating.

## 6.3 Adam (Adaptive Moment Estimation)

- **Two moments**:  
  - **m** (first): running average of gradients (like momentum).  
  - **v** (second): running average of squared gradients (like RMSProp).  
- **Bias correction**: For early steps, m and v are biased toward 0. So we use m_hat = m/(1−β1^t) and v_hat = v/(1−β2^t).  
- **Update**: W -= lr × m_hat / (√v_hat + ε)  
- **Why popular**: Combines momentum (smoother updates) and per-parameter scaling (like RMSProp), and corrects for bias. Works well with little tuning (e.g. β1=0.9, β2=0.999).

**Instructor might ask**: "What are β1 and β2 in Adam?"  
**Answer**: β1 is the decay for the first moment (gradient), β2 for the second moment (squared gradient). Typical values 0.9 and 0.999.

## 6.4 Newton's Method (Our Version: Diagonal Approximation)

- **Classic Newton**: Update = − (Hessian inverse) × gradient. Uses second derivatives.  
- **Full Newton** is expensive (Hessian is huge). We use a **diagonal approximation**:  
  - Approximate "curvature" for each parameter by accumulating **squared gradients** (similar to AdaGrad).  
  - Update: W -= lr × g / (√(accumulated g²) + ε)  
- So we use a **diagonal Hessian** approximation: each parameter has its own step size based on how large its gradients have been.  
- **damping**: We add a small constant so the denominator is never too small (numerical stability).

**In code**: `acc_diag_H_W[i] += gW**2`; then `weights[i] -= lr * gW / (sqrt(acc_diag_H_W[i]) + 1e-8)`.

## 6.5 Muon (Bonus) – High Level

- **Goal**: Optimizer for **2D weight matrices** (e.g. hidden layers).  
- **Idea**:  
  1. Build an update direction using **momentum** (and Nesterov).  
  2. **Orthogonalize** that update matrix using a **Newton–Schulz** iteration (approximates the "nearest orthogonal matrix").  
  3. Use this orthogonalized update to update the weights.  
- **Why**: Orthogonal updates can help training (e.g. better conditioning, less redundant updates).  
- **Output layer**: We use **Adam** for the last layer and for biases, not Muon.

## 6.6 Newton–Schulz (What Muon Uses)

- **Input**: A matrix G (the momentum update).  
- **Output**: A matrix that is "close to orthogonal" (orthonormal columns or rows).  
- **Steps**:  
  1. Normalize G (e.g. by Frobenius norm).  
  2. Apply a fixed iteration 5 times with coefficients (3.4445, −4.7750, 2.0315).  
  3. The iteration is designed so that the singular values of the matrix move toward 1 (orthogonal).  
- **Why 5 steps**: Empirical choice; balances accuracy and cost.  
- **Reference**: Keller Jordan’s Muon blog/post.

**Instructor might ask**: "Why orthogonalize the update?"  
**Answer**: To balance the update across directions (similar to good conditioning). Updates that are almost low-rank can be improved by spreading the update more evenly; orthogonalization does that.

---

# PART 7: TRAINING LOOP (What Happens in train.py)

1. Load data (Digits), split train/test, scale (StandardScaler).  
2. Initialize W1,b1,…,W4,b4 (He initialization).  
3. For each epoch:  
   - Shuffle training data.  
   - For each mini-batch:  
     - **Forward**: `forward_propagation(X_batch, ..., dropout_rate=0.2, training=True)`  
     - **Backward**: `backward_propagation(..., lambda_reg=0.01, dropout_masks=...)`  
     - **Optimizer step**: e.g. `optimizer.step(weights, biases, grads_W, grads_b)`  
   - Record average loss and test accuracy.  
4. After all epochs for one optimizer: plot loss and accuracy vs epoch.  
5. Repeat for each optimizer (GD, RMSProp, Adam, Newton).  
6. Bonus: same for Muon vs Adam, then plot.

---

# PART 8: FILE ROLES

| File | Role |
|------|------|
| **forwardprop.py** | Forward pass: relu, softmax, initialize_parameters, forward_propagation, compute_loss, predict, accuracy. |
| **backprop.py** | Backward pass: backward_propagation (all gradients + loss with L2). |
| **optimizers.py** | GradientDescent, RMSProp, Adam, NewtonMethod, Muon (+ newton_schulz5 for Muon). |
| **train.py** | Load data, train with each optimizer, plot learning curves (interactive). |
| **demo_forward_backward.py** | Demonstrates forward and backward step-by-step to show they work. |
| **REPORT.md** | Comparison of optimizers and Muon vs Adam. |

---

# PART 9: TYPICAL INSTRUCTOR QUESTIONS & SHORT ANSWERS

**Q: What is the difference between forward and backward propagation?**  
A: Forward: input → output (predictions). Backward: from loss backward to compute gradients of the loss w.r.t. every weight and bias using the chain rule.

**Q: Why do we need activation functions?**  
A: Without them, the whole network would be one big linear function. Activations (ReLU, softmax) add non-linearity so the network can learn complex, non-linear patterns.

**Q: What is overfitting and how do we reduce it here?**  
A: Overfitting = model memorizes training data and does poorly on new data. We reduce it with L2 regularization (penalizing large weights) and dropout (randomly dropping neurons during training).

**Q: Why mini-batch and not the full batch?**  
A: Full batch: stable but slow and needs lots of memory. Mini-batch: faster, less memory, and the noise in gradients can help escape bad local minima and generalize better.

**Q: What is the learning rate?**  
A: The step size when we update weights. Too large → unstable or divergence; too small → very slow training.

**Q: Why does Adam often converge faster than plain gradient descent?**  
A: Adam uses momentum (smoother direction) and per-parameter adaptive step sizes (like RMSProp), so it can take larger steps where it’s safe and smaller steps where gradients are large or noisy.

**Q: What is L2 regularization and where do you use it?**  
A: We add (λ/2m) × sum of all W² to the loss. In backprop we add (λ/m)*W to the gradient of each W. We use it for all weight matrices (W1–W4).

**Q: How does dropout work in backward pass?**  
A: We use the **same** dropout mask as in the forward pass. We multiply the incoming gradient by this mask, so dropped neurons get zero gradient and are not updated that step.

**Q: What is the shape of the input and output in your network?**  
A: Input X: (batch_size, 64). Output A4: (batch_size, 10) (probabilities per class). Hidden layers: (m, 128), (m, 64), (m, 32).

**Q: Why softmax for the last layer?**  
A: We do 10-class classification. Softmax turns logits into probabilities that sum to 1, which is the right output for cross-entropy loss.

**Q: What is He initialization and why use it?**  
A: Weights are random with standard deviation √(2/n_in). It keeps the variance of activations roughly stable across layers when using ReLU, so we avoid vanishing/exploding gradients at the start.

**Q: What is Muon and how is it different from Adam?**  
A: Muon is an optimizer for 2D (matrix) weights: it uses momentum and then orthogonalizes the update with a Newton–Schulz iteration. Adam uses first and second moment and no orthogonalization. In our project we use Muon for hidden weight matrices and Adam for the output layer and biases.

**Q: How do you compare the optimizers?**  
A: We train the same network with the same data and hyperparameters with each optimizer (GD, RMSProp, Adam, Newton), then plot training loss and test accuracy vs epoch. We also compare Muon vs Adam the same way.

---

# PART 10: ONE-PAGE CHEAT SHEET (Memorize This)

- **Forward**: Z = A_prev @ W + b, A = ReLU(Z) or softmax(Z). Apply dropout on hidden A in training.  
- **Backward**: Start from dZ4 = A4 − y; then dW = A_prev.T @ dZ + (λ/m)*W, db = sum(dZ); back through ReLU (dZ = dA * (Z>0)) and dropout (dA *= mask).  
- **Loss**: Cross-entropy + (λ/2m)*sum(W²).  
- **GD**: W -= lr * g.  
- **RMSProp**: s = ρ*s + (1−ρ)*g²; W -= lr * g / (√s + ε).  
- **Adam**: m = β1*m + (1−β1)*g, v = β2*v + (1−β2)*g²; m_hat, v_hat; W -= lr * m_hat / (√v_hat + ε).  
- **Newton (ours)**: Accumulate g²; W -= lr * g / (√(accumulated g²) + ε).  
- **Muon**: Momentum update for 2D weights → Newton–Schulz orthogonalize → W -= lr * that; output layer uses Adam.  
- **L2**: Add (λ/m)*W to gradient of W.  
- **Dropout**: Forward: A *= mask (mask has 0s and 1/(1−p)). Backward: dA *= same mask.

---

Good luck with your presentation. If you know Part 1 (terminology), Part 4 (backprop), Part 6 (optimizers), and Part 9 (Q&A), you can answer almost any deep question about this project.
