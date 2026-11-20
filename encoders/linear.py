"""Small NumPy softmax front end; not claimed to be an integer SNN."""
from __future__ import annotations
import numpy as np


def loss_gradient(x, y, weight, bias, l2=0.0):
    logits = x @ weight + bias
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    loss = -np.log(np.maximum(probabilities[np.arange(len(y)), y], 1e-30)).mean()
    loss += 0.5 * l2 * np.square(weight).sum()
    error = probabilities.copy()
    error[np.arange(len(y)), y] -= 1
    error /= len(y)
    return float(loss), x.T @ error + l2 * weight, error.sum(axis=0)


def evaluate(x, y, weight, bias):
    logits = x @ weight + bias
    prediction = logits.argmax(axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(shifted).sum(axis=1))
    nll = np.mean(log_normalizer - shifted[np.arange(len(y)), y])
    return {"accuracy": float(np.mean(prediction == y)), "nll": float(nll),
            "n": len(y)}, prediction


def fit(x, y, valid_x, valid_y, seed, max_epochs=60, batch_size=128,
        learning_rate=0.003, l2=0.001, patience=12):
    """Fixed Adam recipe; checkpoint selection sees validation data only."""
    rng = np.random.default_rng(seed)
    weight = rng.normal(0, 0.01, size=(x.shape[1], 20)).astype(np.float32)
    bias = np.zeros(20, dtype=np.float32)
    mw, vw, mb, vb = np.zeros_like(weight), np.zeros_like(weight), np.zeros_like(bias), np.zeros_like(bias)
    best, best_score, stale, update = None, (-1.0, -float("inf")), 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            ids = order[start:start + batch_size]
            _, gw, gb = loss_gradient(x[ids], y[ids], weight, bias, l2)
            update += 1
            mw = .9 * mw + .1 * gw
            vw = .999 * vw + .001 * gw * gw
            mb = .9 * mb + .1 * gb
            vb = .999 * vb + .001 * gb * gb
            weight -= learning_rate * (mw / (1 - .9**update)) / (np.sqrt(vw / (1 - .999**update)) + 1e-8)
            bias -= learning_rate * (mb / (1 - .9**update)) / (np.sqrt(vb / (1 - .999**update)) + 1e-8)
        metrics, _ = evaluate(valid_x, valid_y, weight, bias)
        history.append({"epoch": epoch, **metrics})
        score = (metrics["accuracy"], -metrics["nll"])
        if score > best_score:
            best_score = score
            best = (weight.copy(), bias.copy(), epoch)
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    assert best is not None
    return best[0], best[1], {"selected_epoch": best[2], "history": history,
                            "epochs_executed": len(history), "optimizer_updates": update}
