"""
microgpt_core.py

Refactored from microgpt.py (@karpathy) for use as a library by a Streamlit
visualization. Model and algorithm are unchanged; the global-state setup is
moved inside a Trainer class so the UI can step training one document at a
time.
"""

import os
import math
import random

random.seed(42)

# =====================================================================
# Hyperparameters (matching microgpt.py)
# =====================================================================
N_LAYER = 1
N_EMBD = 16
BLOCK_SIZE = 16
N_HEAD = 4
HEAD_DIM = N_EMBD // N_HEAD


# =====================================================================
# Value: scalar autograd
# =====================================================================
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data ** other, (self,), (other * self.data ** (other - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other ** -1
    def __rtruediv__(self, other): return other * self ** -1

    def backward(self):
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._children:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad


# =====================================================================
# Dataset and tokenizer
# =====================================================================
def load_dataset(path='input.txt'):
    """Download names.txt if absent and return a shuffled list of strings."""
    if not os.path.exists(path):
        import urllib.request
        url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
        urllib.request.urlretrieve(url, path)
    docs = [line.strip() for line in open(path) if line.strip()]
    random.shuffle(docs)
    return docs


def build_tokenizer(docs):
    uchars = sorted(set(''.join(docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    return uchars, BOS, vocab_size


# =====================================================================
# Model
# =====================================================================
def matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]


def init_state_dict(vocab_size):
    sd = {
        'wte': matrix(vocab_size, N_EMBD),
        'wpe': matrix(BLOCK_SIZE, N_EMBD),
        'lm_head': matrix(vocab_size, N_EMBD),
    }
    for i in range(N_LAYER):
        sd[f'layer{i}.attn_wq'] = matrix(N_EMBD, N_EMBD)
        sd[f'layer{i}.attn_wk'] = matrix(N_EMBD, N_EMBD)
        sd[f'layer{i}.attn_wv'] = matrix(N_EMBD, N_EMBD)
        sd[f'layer{i}.attn_wo'] = matrix(N_EMBD, N_EMBD)
        sd[f'layer{i}.mlp_fc1'] = matrix(4 * N_EMBD, N_EMBD)
        sd[f'layer{i}.mlp_fc2'] = matrix(N_EMBD, 4 * N_EMBD)
    return sd


def get_params(state_dict):
    return [p for mat in state_dict.values() for row in mat for p in row]


def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    max_val = max(v.data for v in logits)
    exps = [(v - max_val).exp() for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def gpt(token_id, pos_id, keys, values, state_dict):
    """Forward one token. Mutates keys/values (KV cache). Returns logits."""
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(N_LAYER):
        # Multi-head attention
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(N_HEAD):
            hs = h * HEAD_DIM
            q_h = q[hs:hs + HEAD_DIM]
            k_h = [ki[hs:hs + HEAD_DIM] for ki in keys[li]]
            v_h = [vi[hs:hs + HEAD_DIM] for vi in values[li]]
            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(HEAD_DIM)) / HEAD_DIM ** 0.5
                for t in range(len(k_h))
            ]
            attn_weights = softmax(attn_logits)
            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(HEAD_DIM)
            ]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        # MLP
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits


# =====================================================================
# Sampling helpers (operate on plain floats, not Values)
# =====================================================================
def apply_sampling(probs, temperature=1.0, top_k=None, top_p=None):
    """Apply temperature, then top-k, then top-p to a list of probabilities.
    Returns a new (renormalized) list of the same length."""
    eps = 1e-12
    # Convert to logits, apply temperature, softmax back
    logits = [math.log(p + eps) for p in probs]
    if temperature != 1.0 and temperature > 0:
        logits = [l / temperature for l in logits]
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    adj = [e / s for e in exps]

    # top-k
    if top_k is not None and 0 < top_k < len(adj):
        ranked = sorted(range(len(adj)), key=lambda i: -adj[i])
        keep = set(ranked[:top_k])
        adj = [p if i in keep else 0.0 for i, p in enumerate(adj)]

    # top-p (nucleus)
    if top_p is not None and 0 < top_p < 1:
        ranked = sorted(range(len(adj)), key=lambda i: -adj[i])
        keep = set()
        cum = 0.0
        for i in ranked:
            keep.add(i)
            cum += adj[i]
            if cum >= top_p:
                break
        adj = [p if i in keep else 0.0 for i, p in enumerate(adj)]

    s = sum(adj)
    if s > 0:
        adj = [p / s for p in adj]
    return adj


# =====================================================================
# Trainer: holds model and steps training one document at a time
# =====================================================================
class Trainer:
    def __init__(self, docs, tokenizer, total_steps=1000, learning_rate=0.01):
        self.docs = docs
        self.uchars, self.BOS, self.vocab_size = tokenizer
        self.state_dict = init_state_dict(self.vocab_size)
        self.params = get_params(self.state_dict)
        # Adam buffers
        self.m = [0.0] * len(self.params)
        self.v = [0.0] * len(self.params)
        # Hyperparameters
        self.learning_rate = learning_rate
        self.beta1 = 0.85
        self.beta2 = 0.99
        self.eps_adam = 1e-8
        self.total_steps = total_steps
        # State
        self.step = 0
        self.losses = []
        self.last_grad_snapshot = None

    def snapshot_weights(self):
        return {k: [[v.data for v in row] for row in mat]
                for k, mat in self.state_dict.items()}

    def snapshot_gradients(self):
        return {k: [[v.grad for v in row] for row in mat]
                for k, mat in self.state_dict.items()}

    def train_step(self):
        doc = self.docs[self.step % len(self.docs)]
        tokens = [self.BOS] + [self.uchars.index(ch) for ch in doc] + [self.BOS]
        n = min(BLOCK_SIZE, len(tokens) - 1)
        keys = [[] for _ in range(N_LAYER)]
        values = [[] for _ in range(N_LAYER)]
        losses = []
        for pos_id in range(n):
            tok_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(tok_id, pos_id, keys, values, self.state_dict)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        loss = (1 / n) * sum(losses)
        loss.backward()
        # Snapshot gradients before zeroing
        self.last_grad_snapshot = self.snapshot_gradients()
        # Adam update
        lr_t = self.learning_rate * max(0.0, 1 - self.step / self.total_steps)
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** (self.step + 1))
            v_hat = self.v[i] / (1 - self.beta2 ** (self.step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + self.eps_adam)
            p.grad = 0
        self.step += 1
        self.losses.append(loss.data)
        return loss.data

    def get_next_token_probs(self, prefix=''):
        """Forward [BOS] + prefix and return next-token probs over the vocabulary."""
        keys = [[] for _ in range(N_LAYER)]
        values = [[] for _ in range(N_LAYER)]
        tokens = [self.BOS]
        for ch in prefix:
            if ch in self.uchars:
                tokens.append(self.uchars.index(ch))
        if len(tokens) > BLOCK_SIZE:
            tokens = tokens[-BLOCK_SIZE:]
        logits = None
        for pos_id, tok_id in enumerate(tokens):
            logits = gpt(tok_id, pos_id, keys, values, self.state_dict)
        probs = softmax(logits)
        return [p.data for p in probs]

    def sample_one(self, temperature=1.0, top_k=None, top_p=None, max_len=None):
        """Generate a single sample (string) from the model."""
        if max_len is None:
            max_len = BLOCK_SIZE
        keys = [[] for _ in range(N_LAYER)]
        values = [[] for _ in range(N_LAYER)]
        token_id = self.BOS
        out = []
        for pos_id in range(max_len):
            logits = gpt(token_id, pos_id, keys, values, self.state_dict)
            probs = softmax(logits)
            raw = [p.data for p in probs]
            adj = apply_sampling(raw, temperature, top_k, top_p)
            if sum(adj) <= 0:
                break
            token_id = random.choices(range(self.vocab_size), weights=adj)[0]
            if token_id == self.BOS:
                break
            out.append(self.uchars[token_id])
        return ''.join(out)
