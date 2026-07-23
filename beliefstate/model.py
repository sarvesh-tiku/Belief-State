"""The agent: activation capture and activation-level steering.

Two backends implement the same interface:

    * :class:`MockAgent` -- a deterministic synthetic transformer whose hidden
      activations carry a *known* ground-truth temporal belief-state.  It lets
      the entire identification + intervention pipeline run offline while giving
      analytically checkable behaviour (the belief direction peaks at a specific
      layer, persists across reasoning steps, differentiates horizons, and
      responds linearly to steering).  This is the testbed the paper's figures
      are generated from when no GPU/model is available.

    * :class:`HFAgent` -- wraps a real Hugging Face causal-LM.  Forward hooks on
      each transformer block capture residual-stream activations, and an
      additive hook implements the steering intervention h' = h + alpha*b
      (Equation 1) at a chosen layer, applied to the reasoning-step token
      positions.

Both return an :class:`AgentRun` holding a ``[n_layers, n_steps, hidden_dim]``
activation tensor aggregated over the tokens of each reasoning step (Section 4:
"activation differences aggregated across tokens associated with reasoning
steps rather than input text").
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .tasks import HORIZONS, PromptSpec


@dataclass
class AgentRun:
    """Captured activations for one prompt (optionally under steering)."""

    activations: np.ndarray  # [n_layers, n_steps, hidden_dim]
    horizon: str
    task_kind: str
    regime: str
    uncertainty: int
    # Final-position next-token logits, when the backend produces them (HF).
    # This is the *behavioural* channel: it is downstream of every steered layer
    # and independent of the belief axis, so a readout taken from it is not the
    # intervention axis re-projected onto itself (the circularity the audit
    # flagged). ``None`` for the mock, which exposes a distinct behavioural axis
    # instead (see MockAgent.behavioral_readout).
    logits: Optional[np.ndarray] = None  # [vocab]

    @property
    def n_layers(self) -> int:
        return self.activations.shape[0]

    @property
    def n_steps(self) -> int:
        return self.activations.shape[1]

    @property
    def hidden_dim(self) -> int:
        return self.activations.shape[2]


class BaseAgent:
    """Common interface for both backends."""

    num_layers: int
    hidden_dim: int

    def run(self, prompt: PromptSpec) -> AgentRun:  # pragma: no cover - abstract
        raise NotImplementedError

    def run_with_steering(
        self, prompt: PromptSpec, layer: int, direction: np.ndarray, alpha: float
    ) -> AgentRun:  # pragma: no cover - abstract
        raise NotImplementedError

    def run_with_patch(
        self, prompt: PromptSpec, layer: int,
        target: np.ndarray, positions: Optional[List[int]] = None,
    ) -> AgentRun:  # pragma: no cover - abstract
        """Replace the residual stream at ``layer`` with ``target`` *during* the
        forward pass, so all downstream layers (and the logits) recompute.

        This is the honest activation patch used by mediation: unlike editing a
        captured tensor after the fact, the substituted activation actually
        propagates, so the behavioural readout reflects the counterfactual.
        ``target`` is ``[hidden_dim]`` (broadcast over the patched positions) or
        ``[n_positions, hidden_dim]``.
        """

        raise NotImplementedError

    def behavioral_readout(self, run: AgentRun) -> float:
        """A scalar behavioural response *independent of the belief axis*.

        For a real model this is a fixed contrast of output logits (how strongly
        the next token leans toward an elevated-risk continuation); it never
        looks at the residual stream, so steering/patching the belief direction
        can only move it *through the model's computation*, not by construction.
        """

        raise NotImplementedError


def _stable_unit_vector(key: str, dim: int) -> np.ndarray:
    """A deterministic unit vector seeded by a string key."""

    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return v / (np.linalg.norm(v) + 1e-12)


# ===========================================================================
# Mock backend
# ===========================================================================

class MockAgent(BaseAgent):
    """Synthetic transformer with an embedded ground-truth belief-state.

    Activation model (per layer L, reasoning step s):

        h[L, s] =  gate_h(L; horizon) * persist(s) * e * b[horizon]     (belief)
                 + surface_scale * u_surface(prompt_surface)            (nuisance)
                 + eps                                                  (noise)

    where ``e`` is a scalar "future downside expectation" derived from the
    volatility regime of the historical context and the uncertainty of the
    disclosure, ``b[horizon]`` are fixed near/medium belief directions with
    partial overlap, ``gate_h`` is a horizon-dependent Gaussian bump over layers
    (peak = mock_belief_layer, shifted later for the medium horizon), and
    ``persist`` decays slowly across reasoning steps.  Surface-only changes move
    only the ``u_surface`` term, which is (near-)orthogonal to ``b`` -- so null
    contrasts produce small activation differences that do not load on the
    belief directions.
    """

    def __init__(self, cfg, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        m = cfg.model
        self.hidden_dim = int(m.mock_hidden_dim)
        self.num_layers = int(m.mock_num_layers)
        self.belief_layer = int(m.mock_belief_layer)
        self.noise = float(m.mock_noise)
        self.rng = rng or np.random.default_rng(cfg.seed + 7)

        # Fixed ground-truth belief directions with partial overlap between
        # horizons (the paper reports *partial but reproducible* separation).
        base = _stable_unit_vector("belief-base", self.hidden_dim)
        near_axis = _stable_unit_vector("belief-near", self.hidden_dim)
        med_axis = _stable_unit_vector("belief-medium", self.hidden_dim)
        # Share ~55% with a common base -> cosine(b_near, b_medium) ~ 0.5.
        b_near = 0.75 * base + 0.66 * near_axis
        b_med = 0.75 * base + 0.66 * med_axis
        self._b = {
            "near_term": b_near / np.linalg.norm(b_near),
            "medium_term": b_med / np.linalg.norm(b_med),
        }
        # Horizon-dependent peak layers (Fig 1A: near peaks earlier than medium).
        self._peak = {
            "near_term": self.belief_layer,
            "medium_term": min(self.num_layers, self.belief_layer + 3),
        }
        self._width = 2.5

        # Behavioural axis: what the agent would *output*. It shares direction
        # with the belief axis only partially (cosine ~0.6), so a readout along
        # it is not the belief projection re-labelled -- steering/patching the
        # belief moves behaviour only through this partial coupling, exactly the
        # non-circular readout the audit demanded. The readout uses the *last*
        # layer (downstream of every steered/patched layer).
        w_raw = 0.6 * self._b["near_term"] + 0.8 * _stable_unit_vector(
            "behavior-axis", self.hidden_dim
        )
        self._w_behavior = w_raw / np.linalg.norm(w_raw)
        self._behavior_gain = 1.5

    # -- ground-truth accessors (used by tests / analytical checks) ---------

    def true_direction(self, horizon: str) -> np.ndarray:
        return self._b[horizon].copy()

    def peak_layer(self, horizon: str) -> int:
        return self._peak[horizon]

    # -- helpers ------------------------------------------------------------

    def _gate(self, horizon: str) -> np.ndarray:
        layers = np.arange(self.num_layers + 1)
        peak = self._peak[horizon]
        g = np.exp(-0.5 * ((layers - peak) / self._width) ** 2)
        return g

    @staticmethod
    def _persist(n_steps: int) -> np.ndarray:
        s = np.arange(n_steps)
        # High and slowly decaying -> matches Fig 1C belief curve.
        return 0.95 * np.exp(-0.03 * s) + 0.02 * np.cos(0.5 * s)

    def _expectation(self, prompt: PromptSpec) -> float:
        """Scalar future downside expectation from regime + uncertainty."""

        r = 1.0 if prompt.regime == "high_vol" else -1.0
        u = float(prompt.uncertainty - 1)  # {-1, 0, 1}
        return 0.7 * r + 0.5 * u

    def _surface_vector(self, prompt: PromptSpec) -> np.ndarray:
        """Nuisance direction driven only by surface form.

        Depends on the variant surface axes and any benign formatting markers,
        NOT on the semantic (regime/uncertainty) content.  This is what null
        contrasts perturb.
        """

        v = prompt.variant
        # A signature capturing surface form: variant axes + whitespace/header
        # idiosyncrasies of the rendered text, but not the semantic fields.
        header_sig = "".join(ch for ch in prompt.text if ch in "[]{} \t").__len__()
        sig = f"{v.key()}|hdr={header_sig}"
        return _stable_unit_vector(sig, self.hidden_dim)

    def _base_activations(self, prompt: PromptSpec) -> np.ndarray:
        n_steps = prompt.n_steps
        n_layers = self.num_layers + 1
        d = self.hidden_dim

        e = self._expectation(prompt)
        b = self._b[prompt.horizon]
        gate = self._gate(prompt.horizon)          # [n_layers]
        persist = self._persist(n_steps)           # [n_steps]
        u_surface = self._surface_vector(prompt)

        # Deterministic per-instance jitter (stable across steering calls for the
        # same prompt) so PCA clouds look realistic but runs are reproducible.
        inst_seed = int.from_bytes(
            hashlib.sha256(
                f"{prompt.firm_id}|{prompt.period}|{prompt.horizon}|"
                f"{prompt.task_kind}|{prompt.variant.key()}".encode()
            ).digest()[:8],
            "little",
        )
        inst_rng = np.random.default_rng(inst_seed)
        jitter = 0.15 * inst_rng.standard_normal()

        belief_scale = (gate[:, None] * persist[None, :])  # [L, S]
        # belief term: outer over (L,S) scalar field times b
        acts = belief_scale[:, :, None] * ((e + jitter) * b)[None, None, :]

        # surface nuisance: layer-flat, small, (near-)orthogonal to b
        surf_scale = 0.25
        acts = acts + surf_scale * u_surface[None, None, :]

        # additive noise
        acts = acts + self.noise * 0.15 * inst_rng.standard_normal(
            (n_layers, n_steps, d)
        )
        return acts

    # -- interface ----------------------------------------------------------

    def run(self, prompt: PromptSpec) -> AgentRun:
        acts = self._base_activations(prompt)
        return AgentRun(
            activations=acts,
            horizon=prompt.horizon,
            task_kind=prompt.task_kind,
            regime=prompt.regime,
            uncertainty=prompt.uncertainty,
        )

    def run_with_steering(
        self, prompt: PromptSpec, layer: int, direction: np.ndarray, alpha: float
    ) -> AgentRun:
        acts = self._base_activations(prompt)
        unit = direction / (np.linalg.norm(direction) + 1e-12)
        # Apply h' = h + alpha*b at the chosen layer, across reasoning-step
        # positions (interventions applied during intermediate reasoning steps).
        # The edit propagates to downstream layers via _propagate_belief so the
        # behavioural readout (last-third layers) actually moves -- otherwise a
        # steer at the peak layer would leave the readout untouched, which is not
        # how a real forward pass behaves.
        delta = alpha * unit[None, :]  # [1, d] broadcast over steps
        acts[layer, :, :] = acts[layer, :, :] + delta
        self._propagate_delta(acts, prompt.horizon, layer,
                              np.broadcast_to(delta, (acts.shape[1], self.hidden_dim)))
        return AgentRun(
            activations=acts,
            horizon=prompt.horizon,
            task_kind=prompt.task_kind,
            regime=prompt.regime,
            uncertainty=prompt.uncertainty,
        )

    def run_with_patch(
        self, prompt: PromptSpec, layer: int,
        target: np.ndarray, positions: Optional[List[int]] = None,
    ) -> AgentRun:
        """Replace the residual at ``layer`` with ``target`` and propagate.

        ``target`` is ``[hidden_dim]`` or ``[n_steps, hidden_dim]``. The mock
        has no learned weights, so downstream propagation is *simulated*: the
        belief-axis component of the (target - original) change at ``layer`` is
        pushed to every downstream layer, scaled by the mock's layer gate (the
        analogue of the real model's weights), leaving surface + noise intact.
        Because only the *delta* propagates, patching to a layer's own value is
        exactly the identity. Positions are accepted for interface parity with
        HFAgent but ignored (the mock pools reasoning steps).
        """

        acts = self._base_activations(prompt)
        target = np.asarray(target, dtype=float)
        if target.ndim == 1:
            target = np.broadcast_to(target, (prompt.n_steps, self.hidden_dim))
        delta = target - acts[layer, :, :]      # [n_steps, d]
        acts[layer, :, :] = target
        self._propagate_delta(acts, prompt.horizon, layer, delta)
        return AgentRun(
            activations=acts,
            horizon=prompt.horizon,
            task_kind=prompt.task_kind,
            regime=prompt.regime,
            uncertainty=prompt.uncertainty,
        )

    def _propagate_delta(
        self, acts: np.ndarray, horizon: str, from_layer: int, delta: np.ndarray
    ) -> None:
        """Propagate the belief-axis part of an edit at ``from_layer`` downstream.

        The mock's belief content at layer L is ``gate[L]*persist[s]*e*b``, so a
        change ``dp`` in the belief projection at ``from_layer`` corresponds to a
        change ``dp * gate[L]/gate[from_layer]`` at layer L. We add exactly that
        along ``b`` for every layer L > from_layer, preserving each layer's
        surface + noise (and leaving the off-belief part of the edit local to
        ``from_layer``, as a real block's residual add would be). This keeps a
        self-patch as the identity while making belief edits causally visible to
        the downstream readout.
        """

        b = self._b[horizon]
        gate = self._gate(horizon)
        g0 = gate[from_layer]
        if abs(g0) < 1e-9:
            return
        n_steps = acts.shape[1]
        for s in range(n_steps):
            dp = float(delta[s] @ b)  # change in belief projection at from_layer
            if abs(dp) < 1e-12:
                continue
            for L in range(from_layer + 1, acts.shape[0]):
                acts[L, s] = acts[L, s] + (dp * gate[L] / g0) * b

    def _readout_layers(self) -> range:
        """Layers used for the behavioural readout: strictly downstream of the
        near-horizon peak, so it never coincides with the intervention layer."""

        start = max(self.belief_layer + 1, int(0.66 * (self.num_layers + 1)))
        start = min(start, self.num_layers)  # keep at least the last layer
        return range(start, self.num_layers + 1)

    def behavioral_readout(self, run: AgentRun) -> float:
        """Scalar behaviour along the behaviour axis, read from the downstream
        layers only. Distinct axis + distinct layers from the belief readout, so
        it is not the intervention projection re-labelled."""

        rows = self._readout_layers()
        pooled = np.mean([run.activations[L].mean(axis=0) for L in rows], axis=0)
        return self._behavior_gain * float(pooled @ self._w_behavior)


# ===========================================================================
# Hugging Face backend
# ===========================================================================

class SpanAlignmentError(RuntimeError):
    """Raised when reasoning-step markers cannot be aligned to token spans."""


class HFAgent(BaseAgent):
    """Wraps a real HF causal-LM with activation capture + steering hooks.

    Layer convention (verified empirically on gpt2, and the standard HF
    contract): ``output.hidden_states`` is a tuple of length ``num_layers + 1``
    where ``hidden_states[0]`` is the token+position embedding and
    ``hidden_states[L]`` (L >= 1) is the residual stream *after* block ``L-1``.
    We keep this indexing end to end, so a captured/analyzed layer index ``L``
    and a steered layer index ``L`` refer to the *same* residual stream:

        * to read layer ``L`` we take ``hidden_states[L]``;
        * to write layer ``L`` (L >= 1) we hook ``blocks[L-1]`` and add to its
          output; layer 0 (embeddings) is steered with a pre-hook on
          ``blocks[0]``.

    This removes the off-by-one that previously made identification and steering
    act on different layers.
    """

    def __init__(self, cfg):
        import torch  # local import so the mock path needs no torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.torch = torch
        # Deterministic seeding of torch global RNG (defensive; forward passes
        # under eval() + no_grad are deterministic, but generation is not).
        torch.manual_seed(int(cfg.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg.seed))

        name = cfg.model.hf_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Right padding keeps real tokens left-aligned, so per-text character
        # offsets map to token indices unchanged under batching.
        self.tokenizer.padding_side = "right"

        self.device = self._select_device()
        self.model = AutoModelForCausalLM.from_pretrained(
            name, output_hidden_states=True
        )
        self.model.eval()
        self.model.to(self.device)
        self.hidden_dim = int(self.model.config.hidden_size)
        self.num_layers = int(self.model.config.num_hidden_layers)
        self._blocks = self._locate_blocks()
        if len(self._blocks) != self.num_layers:
            raise RuntimeError(
                f"located {len(self._blocks)} blocks but config reports "
                f"{self.num_layers} layers; check hf_block_path"
            )
        self._behavior_tokens = self._build_behavior_tokens()

    def _build_behavior_tokens(self) -> dict:
        """Token ids for the behavioural logit contrast.

        The behavioural readout scores how strongly the model's next-token
        distribution leans toward an elevated-risk continuation versus a benign
        one. We use small lexical sets and average their logits; a leading space
        is prepended because GPT-2-style BPE encodes mid-sentence words with a
        preceding space. Tokens that split into multiple pieces contribute their
        first piece (a reasonable proxy for the initial next-token decision).
        """

        risk_words = ["risk", "loss", "decline", "downside", "volatile", "weak"]
        calm_words = ["stable", "steady", "strong", "gain", "growth", "safe"]

        def ids(words: List[str]) -> List[int]:
            out: List[int] = []
            for w in words:
                for form in (" " + w, w):
                    toks = self.tokenizer.encode(form, add_special_tokens=False)
                    if toks:
                        out.append(int(toks[0]))
            return sorted(set(out))

        return {"risk": ids(risk_words), "calm": ids(calm_words)}

    def _select_device(self):
        torch = self.torch
        pref = getattr(self.cfg.model, "hf_device", "auto")
        if pref and pref != "auto":
            return torch.device(pref)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _locate_blocks(self) -> list:
        """Locate the transformer block modules (nn.ModuleList) for hooks.

        Honors an explicit ``model.hf_block_path`` config override, else tries
        the common layouts (GPT-2, Llama/Mistral, GPT-NeoX, GPT-J, OPT, Falcon).
        """

        override = getattr(self.cfg.model, "hf_block_path", None)
        candidates = [override] if override else []
        candidates += [
            "transformer.h",          # GPT-2, GPT-J
            "model.layers",           # Llama, Mistral, Qwen
            "gpt_neox.layers",        # GPT-NeoX, Pythia
            "model.decoder.layers",   # OPT
            "transformer.blocks",     # MPT
        ]
        for path in candidates:
            if not path:
                continue
            obj = self.model
            ok = True
            for part in path.split("."):
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    ok = False
                    break
            if ok:
                return list(obj)
        raise RuntimeError(
            "Could not locate transformer blocks for hooks; set "
            "model.hf_block_path in config for this architecture."
        )

    # -- token-span alignment ----------------------------------------------

    def _step_token_spans(self, text: str, n_steps: int,
                          offsets: List[tuple]) -> List[range]:
        """Token spans for each reasoning-step marker ``<stepK>``.

        Aligns activations to the tokens of each reasoning step (the text
        between consecutive markers), rather than the input text (Section 4).
        Markers are located by character offset, which is robust to the marker
        tokenizing into several subword tokens. Raises SpanAlignmentError on any
        missing marker or empty/degenerate span rather than silently collapsing
        to the last token, so alignment bugs surface instead of corrupting the
        activations.
        """

        marker_char = []
        for i in range(n_steps):
            idx = text.find(f"<step{i}>")
            if idx < 0:
                raise SpanAlignmentError(
                    f"reasoning-step marker <step{i}> not found in prompt text"
                )
            marker_char.append(idx)
        if marker_char != sorted(marker_char):
            raise SpanAlignmentError("reasoning-step markers are out of order")
        bounds = marker_char + [len(text)]

        spans: List[range] = []
        for i in range(n_steps):
            start_char, end_char = bounds[i], bounds[i + 1]
            tok_idx = [
                j for j, (a, b) in enumerate(offsets)
                if a >= start_char and b <= end_char and b > a
            ]
            if not tok_idx:
                raise SpanAlignmentError(
                    f"reasoning step {i} maps to zero tokens "
                    f"(chars [{start_char},{end_char}))"
                )
            spans.append(range(tok_idx[0], tok_idx[-1] + 1))
        # Guard against overlap between consecutive steps.
        for i in range(1, n_steps):
            if spans[i].start < spans[i - 1].stop:
                raise SpanAlignmentError(
                    f"reasoning-step spans {i - 1} and {i} overlap"
                )
        return spans

    # -- steering hook builder ---------------------------------------------

    def _steer_handles(self, layer: int, unit_t, alpha: float,
                       positions: List[int]) -> list:
        """Register the additive steering hook h' = h + alpha*unit at ``layer``.

        For layer >= 1 we add to the output of ``blocks[layer-1]`` (== residual
        stream ``hidden_states[layer]``). For layer 0 we add to the *input* of
        ``blocks[0]`` (== the embedding stream) via a forward pre-hook.
        """

        pos = [p for p in positions]

        def out_hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            add = (alpha * unit_t).to(hs.dtype)
            for p in pos:
                if p < hs.shape[1]:
                    hs[0, p, :] = hs[0, p, :] + add
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

        def pre_hook(module, args, kwargs):
            hs = args[0]
            add = (alpha * unit_t).to(hs.dtype)
            for p in pos:
                if p < hs.shape[1]:
                    hs[0, p, :] = hs[0, p, :] + add
            return (hs,) + tuple(args[1:]), kwargs

        if layer == 0:
            return [self._blocks[0].register_forward_pre_hook(
                pre_hook, with_kwargs=True)]
        if not 1 <= layer <= self.num_layers:
            raise ValueError(
                f"steering layer {layer} out of range [0, {self.num_layers}]"
            )
        return [self._blocks[layer - 1].register_forward_hook(out_hook)]

    # -- capture ------------------------------------------------------------

    def _capture(self, prompt: PromptSpec, steer=None, patch=None):
        """Run one forward pass; return ``(activations[L,S,d], logits[vocab])``.

        ``steer`` = ``(layer, unit, alpha)`` adds ``alpha*unit`` at ``layer``.
        ``patch`` = ``(layer, target_2d)`` *replaces* the residual at ``layer``
        on the reasoning-step positions with ``target_2d`` ([n_steps, d]); the
        replacement propagates through the rest of the network, so the returned
        logits reflect the counterfactual.
        """

        torch = self.torch
        enc = self.tokenizer(prompt.text, return_tensors="pt",
                             return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        spans = self._step_token_spans(prompt.text, prompt.n_steps, offsets)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        handles = []
        if steer is not None:
            layer, unit, alpha = steer
            unit_t = torch.tensor(unit, dtype=torch.float32, device=self.device)
            steer_positions = sorted({p for span in spans for p in span})
            handles = self._steer_handles(layer, unit_t, alpha, steer_positions)
        if patch is not None:
            p_layer, target_2d = patch
            target_t = torch.tensor(target_2d, dtype=torch.float32,
                                    device=self.device)
            handles += self._patch_handles(p_layer, target_t, spans)

        try:
            with torch.no_grad():
                out = self.model(**enc)
        finally:
            for h in handles:
                h.remove()

        acts = self._aggregate(out.hidden_states, [spans], prompt.n_steps)[0]
        logits = out.logits[0, -1, :].detach().cpu().float().numpy()
        return acts, logits

    def _patch_handles(self, layer: int, target_t, spans: List[range]) -> list:
        """Register a hook that patches the residual at ``layer`` so each
        reasoning step's *pooled* value becomes ``target_t[s]``.

        ``target_t[s]`` is a per-step target for the mean-pooled residual (that
        is how activations are captured). We apply it as an *additive delta*:
        for step ``s`` with span tokens ``P``, we add ``target_t[s] - mean_P(h)``
        to every token in ``P``. This makes the step's new pooled value exactly
        ``target_t[s]`` while preserving within-step token variation -- so
        patching a step with its own captured value is exactly the identity, and
        the mediator swap shifts only by the intended per-step amount.

        Mirrors the read/write layer convention of the steering hooks: layer 0
        is patched as the input of ``blocks[0]``; layer L>=1 as the output of
        ``blocks[L-1]`` (== ``hidden_states[L]``).
        """

        n_rows = target_t.shape[0]
        span_tokens = [
            [p for p in span] for span in spans
        ]

        def apply(hs):
            for s, toks in enumerate(span_tokens):
                if s >= n_rows:
                    continue
                idx = [p for p in toks if p < hs.shape[1]]
                if not idx:
                    continue
                current_mean = hs[0, idx, :].mean(dim=0)
                delta = (target_t[s].to(hs.dtype) - current_mean)
                for p in idx:
                    hs[0, p, :] = hs[0, p, :] + delta
            return hs

        def out_hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            hs = apply(hs)
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

        def pre_hook(module, args, kwargs):
            hs = apply(args[0])
            return (hs,) + tuple(args[1:]), kwargs

        if layer == 0:
            return [self._blocks[0].register_forward_pre_hook(
                pre_hook, with_kwargs=True)]
        if not 1 <= layer <= self.num_layers:
            raise ValueError(
                f"patch layer {layer} out of range [0, {self.num_layers}]"
            )
        return [self._blocks[layer - 1].register_forward_hook(out_hook)]

    @staticmethod
    def _aggregate(hidden_states, spans_per_example, n_steps: int) -> np.ndarray:
        """Mean-pool residual-stream activations over each step's token span.

        Returns ``[batch, n_layers+1, n_steps, d]``.
        """

        n_hs = len(hidden_states)
        d = hidden_states[0].shape[-1]
        batch = hidden_states[0].shape[0]
        acts = np.zeros((batch, n_hs, n_steps, d), dtype=np.float32)
        for L in range(n_hs):
            layer_hs = hidden_states[L].detach().cpu().float().numpy()  # [B,seq,d]
            for e in range(batch):
                for s, span in enumerate(spans_per_example[e]):
                    idx = [p for p in span if p < layer_hs.shape[1]]
                    if idx:
                        acts[e, L, s] = layer_hs[e, idx].mean(axis=0)
        return acts

    # -- interface ----------------------------------------------------------

    def run(self, prompt: PromptSpec) -> AgentRun:
        acts, logits = self._capture(prompt)
        return AgentRun(acts, prompt.horizon, prompt.task_kind,
                        prompt.regime, prompt.uncertainty, logits=logits)

    def run_with_steering(
        self, prompt: PromptSpec, layer: int, direction: np.ndarray, alpha: float
    ) -> AgentRun:
        unit = direction / (np.linalg.norm(direction) + 1e-12)
        acts, logits = self._capture(prompt, steer=(layer, unit, alpha))
        return AgentRun(acts, prompt.horizon, prompt.task_kind,
                        prompt.regime, prompt.uncertainty, logits=logits)

    def run_with_patch(
        self, prompt: PromptSpec, layer: int,
        target: np.ndarray, positions: Optional[List[int]] = None,
    ) -> AgentRun:
        target = np.asarray(target, dtype=float)
        if target.ndim == 1:
            target = np.broadcast_to(target, (prompt.n_steps, self.hidden_dim))
        acts, logits = self._capture(prompt, patch=(layer, target))
        return AgentRun(acts, prompt.horizon, prompt.task_kind,
                        prompt.regime, prompt.uncertainty, logits=logits)

    def behavioral_readout(self, run: AgentRun) -> float:
        """Risk-minus-calm next-token log-probability contrast.

        Reads only the output logits, so it is independent of the belief axis
        and strictly downstream of every steered/patched layer. Returns the
        difference of mean log-softmax mass on the risk vs calm token sets.
        """

        if run.logits is None:
            raise ValueError("behavioral_readout requires logits (HF backend)")
        z = run.logits.astype(np.float64)
        z = z - z.max()
        logZ = np.log(np.exp(z).sum())
        logp = z - logZ
        risk = float(np.mean([logp[i] for i in self._behavior_tokens["risk"]]))
        calm = float(np.mean([logp[i] for i in self._behavior_tokens["calm"]]))
        return risk - calm

    def batch_run(self, prompts: List[PromptSpec]) -> List[AgentRun]:
        """Capture activations for several prompts in one padded forward pass.

        Steering is intentionally not supported here (per-row hooks differ); use
        :meth:`run_with_steering` for interventions. Right padding keeps real
        tokens left-aligned so per-text offsets stay valid.
        """

        if not prompts:
            return []
        torch = self.torch
        texts = [p.text for p in prompts]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True,
                             return_offsets_mapping=True)
        offsets_batch = enc.pop("offset_mapping").tolist()
        spans_per_example = [
            self._step_token_spans(p.text, p.n_steps, offsets_batch[e])
            for e, p in enumerate(prompts)
        ]
        n_steps = prompts[0].n_steps
        attn = enc["attention_mask"]
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        acts = self._aggregate(out.hidden_states, spans_per_example, n_steps)
        # Right padding: the last *real* token is at (row length - 1), not -1.
        last_real = attn.sum(dim=1).long() - 1  # [B]
        all_logits = out.logits.detach().cpu().float().numpy()
        return [
            AgentRun(acts[e], p.horizon, p.task_kind, p.regime, p.uncertainty,
                     logits=all_logits[e, int(last_real[e])])
            for e, p in enumerate(prompts)
        ]


def build_agent(cfg, rng: Optional[np.random.Generator] = None) -> BaseAgent:
    """Factory selecting the backend from config."""

    backend = cfg.model.backend
    if backend == "mock":
        return MockAgent(cfg, rng=rng)
    if backend == "hf":
        return HFAgent(cfg)
    raise ValueError(f"unknown model.backend: {backend}")
