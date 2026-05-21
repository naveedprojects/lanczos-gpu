# Outreach to Prof. Borovitskiy — message draft

This is a starting-point draft, **not** the final wording — adjust to your voice. The aim is short (~6-8 lines), lands the headline in the first sentence, and links to the artifacts.

---

## LinkedIn message — short form (~6-8 lines)

> Hi Viacheslav,
>
> Coming back on the differentiable Lanczos work — sorry for the long silence, I wanted to come back with something concrete. Your Table 2 footnote on the SS-IMGP page-10 limitation (GPyTorch Lanczos quality forcing the fall-back to dense `torch.linalg.eigh`) was the limitation I've been fixing.
>
> I've built a differentiable stochastic Lanczos quadrature with the Krämer 2024 adjoint, batched over Hutchinson probes, with end-to-end gradients through arbitrary parameterised matvecs in PyTorch. On a single RTX 3080 Ti Laptop:
>
> - At your MR-MNIST scale (N = 10⁵), naive BBMM-style Lanczos diverges to NaN, ours converges with identical loss / hyperparameter trajectories to the dense baseline.
> - At N = 10⁶ — a scale where neither dense `slogdet` nor naive BBMM is usable — ours runs a full marginal-likelihood maximisation in about 90 seconds.
> - At N = 500K and 1M, naive Lanczos diverges to NaN on iter 0; ours scales smoothly.
>
> Code, tests, and a 4-page writeup are at github.com/naveedprojects/lanczos-gpu. The headline figure is [figures/imgp_scaling.png](figures/imgp_scaling.png).
>
> Would love to know if this is useful for what you're working on now, and to hear your thoughts. Happy to walk through the code or rerun on a different graph / scale if helpful.
>
> Best regards,
> Naveed

---

## Longer technical version (if you'd rather email)

(Two paragraphs — keeps the same content but with more context for someone less time-pressured.)

> Hi Viacheslav,
>
> Following up on our LinkedIn exchange from March — apologies for the long silence on my end. I wanted to come back with something more substantial than just an update.
>
> Your IMGP paper's Table 2 footnote (page 10) names a specific limitation: GPyTorch's BBMM Lanczos quality wasn't enough for SS-IMGP, forcing the fall-back to dense `torch.linalg.eigh`, which doesn't scale beyond ~10⁴ nodes. That's the gap I've been working on.
>
> What I built: a differentiable Stochastic Lanczos Quadrature for log-determinants, with custom-backward via the Krämer 2024 adjoint formulas (transcribed into PyTorch, all Hutchinson probes batched into a single Lanczos sweep over a (n, m_probes) matrix), end-to-end differentiable through arbitrary parameterised matvecs. Designed to match the exact place in your training loop where `precision_operator.inv_quad_logdet(logdet=True)[1]` is called.
>
> Concrete results on a single RTX 3080 Ti Laptop (16 GB):
>
> - At N = 100 000 (= MR-MNIST), the naive BBMM-style Lanczos (no reortho, unrolled autograd through the Lanczos iterations — the row of your Table 2 the footnote documents as failing) converges in this prototype, but at N = 500 000 and N = 1 000 000 it diverges to NaN on iteration 0. Ours converges throughout.
> - The dense `slogdet` baseline (the SS-IMGP-full row) truncates at N = 2 000 on 16 GB. At that N it's already 27× slower than ours; past that, uncomputable.
> - At N = 10⁶, ours runs a full marginal-likelihood maximisation in about 90 seconds. Memory and wall-clock both leave generous headroom; the toolkit is not at its scaling limit.
>
> Code, tests, demo, and a 4-6 page writeup are at github.com/naveedprojects/lanczos-gpu. The 4-panel headline figure is in `figures/imgp_scaling.png`; the writeup is `docs/WRITEUP.md`.
>
> I'd love to know if this is useful for what you're working on currently. Two specific things I'm curious about:
>
> 1. Whether you'd be interested in trying it on a real IMGP dataset (the synthetic graphs I tested on were structurally identical but not the actual MR-MNIST setup).
> 2. Where the natural next direction is — the differentiable SLQ machinery generalises to any matrix function (sqrt, exp, resolvent), so it can drop into spectral graph filters, BNN Laplace approximations, etc.
>
> Happy to walk through the code in detail, or rerun on whatever dataset you have lying around.
>
> Best regards,
> Naveed

---

## What to send with the message

- Link to the GitHub repo (after you push the current state).
- Link to `figures/imgp_scaling.png` (or attach the PNG directly — LinkedIn supports inline images).
- Optionally, `docs/WRITEUP.md` as a PDF if you want the formal-paper feel. Use `pandoc docs/WRITEUP.md -o writeup.pdf --pdf-engine=xelatex` or similar.

## Tone calibration

The original tone of your exchanges was direct and technically dense — short paragraphs, concrete numbers, no hedging. The drafts above try to match that. Avoid:

- "I hope this finds you well" / similar opening filler. Skip straight to content.
- Apologising more than once for the gap. One acknowledgment, then move on.
- Asking for feedback in a way that implies you need validation — let the numbers speak.

What lands with a technical reader: a specific page-and-table reference to *his own paper* (Table 2, page 10, the footnote), then a numeric comparison, then the artifact.

## What NOT to do

- Don't lead with the *math* (Krämer adjoint, restart-aware adjoints, etc.). The reader's first question is "does it work," not "is the derivation new."
- Don't oversell the novelty of the underlying adjoint formulas — those are Krämer 2024. The contribution is the *production-quality artifact* at his exact problem scale.
- Don't include the LinkedIn dialogue history. He'll remember.
