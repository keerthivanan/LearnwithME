# Reasoning Models, Multimodal LLMs & Scaling Laws

> o1, DeepSeek-R1, vision LLMs, and why bigger = better (until it doesn't).
> Critical topics missing from every basic GenAI curriculum.

---

## PART 1: REASONING MODELS — O1, O3, DEEPSEEK-R1

### Why Standard LLMs Fail at Reasoning

Standard LLMs predict the next token. Fast, fluent, but:
```
Q: "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball.
    How much does the ball cost?"

GPT-3.5 (fast answer): "$0.10"   ← WRONG (common cognitive bias)
Correct answer: $0.05 (bat=$1.05, ball=$0.05, difference=$1.00 ✓)
```

The model gives the intuitive wrong answer because it was trained to be fast,
not to reason carefully.

**Chain-of-Thought helps but has limits.** You can prompt "think step by step"
but the model wasn't trained to explore multiple reasoning paths, backtrack,
or verify its own work.

---

### OpenAI o1 — Process Reward + RL for Reasoning

**Released:** September 2024
**Core idea:** Train the model to spend compute at **inference time** doing extended reasoning before answering.

#### How o1 Was Trained (What We Know)

**Step 1: Collect long reasoning chains**
Generate many long chains-of-thought (CoT) for hard problems (math, coding, logic).
Use human experts to label which reasoning steps are **correct** vs **incorrect**.

**Step 2: Train a Process Reward Model (PRM)**
```
Standard reward model: scores the FINAL answer (correct/incorrect)
Process reward model:  scores each STEP in the reasoning chain

PRM("Step: Let x be the ball price. Then bat = x + 1.00...") → 0.9 (good step)
PRM("Step: The ball costs $0.10 because...") → 0.1 (bad step)
```

This is the key innovation: reward **correct reasoning steps**, not just correct answers.

**Step 3: RL with the PRM as reward**
Use reinforcement learning where each reasoning step gets a reward from the PRM.
The model learns to:
- Generate correct intermediate steps
- Backtrack when a step seems wrong
- Try multiple approaches (like a human thinking hard)
- Verify its answer before committing

**Step 4: Inference = extended thinking**
At test time, o1 generates a long "thinking" process (hidden from user) before answering.
More thinking tokens = better accuracy on hard problems.

```
User: "Solve this integral: ∫ x²e^x dx"

o1 internal thinking (hidden):
  "Let me try integration by parts. u = x², dv = e^x dx
   Then du = 2x dx, v = e^x
   ∫ x²e^x dx = x²e^x - ∫ 2x e^x dx
   Now I need ∫ 2x e^x dx, use parts again...
   [20 more reasoning steps]
   Final: (x²-2x+2)e^x + C
   Let me verify by differentiating... ✓"

o1 output: "(x²-2x+2)e^x + C"
```

#### o1 Performance

- AIME (American Invitational Mathematics Exam): 74% (vs GPT-4: 13%)
- Competition math (MATH benchmark): 94% (vs GPT-4: 72%)
- PhD-level science questions (GPQA): 78% (vs GPT-4: 53%)

The cost: o1 uses 10-100× more tokens per query than GPT-4.
Slower, more expensive, but dramatically better on hard reasoning tasks.

---

### DeepSeek-R1 — Fully Open Reasoning Model

**Released:** January 2025
**Significance:** First open-source reasoning model matching o1. Fully transparent training.

#### The Training Pipeline (4 Stages)

**Stage 1: Cold Start with Long CoT Data**
```
Problem: RL from scratch produces incoherent reasoning
Solution: First collect or generate ~thousands of long CoT examples
          Fine-tune the base model on these (SFT)
          Now the model knows how to format long reasoning
```

**Stage 2: GRPO Training (Reinforcement Learning)**

GRPO = Group Relative Policy Optimization

Instead of PPO (which needs a separate value/critic network), GRPO:
1. For each problem, sample G responses from the current policy
2. Compute reward for each response (rule-based: is the answer correct?)
3. Use the group's average reward as the baseline
4. Update policy to increase probability of above-average responses

```
G=8 responses to "What is 15% of 340?"
  Response 1: "51" (correct) → reward = 1
  Response 2: "51" (correct) → reward = 1
  Response 3: "51" (correct) → reward = 1
  Response 4: "52" (wrong)   → reward = 0
  Response 5: "51" (correct) → reward = 1
  ...
  Group average reward = 0.625
  
Update policy: increase prob of correct responses,
               decrease prob of incorrect ones
```

**Why GRPO over PPO?**
- No separate critic network (saves memory — PPO needs 2× model parameters)
- More stable training
- Rule-based rewards (no reward model to train)
- Works well for verifiable domains (math, coding)

**Stage 3: Rejection Sampling Fine-tuning**
- Generate many responses using the RL-trained model
- Keep only high-quality responses (correct + clear reasoning)
- Fine-tune the model on these filtered high-quality examples

**Stage 4: Final RLHF (for helpfulness + safety)**
- Standard DPO/RLHF to make the model helpful and safe
- Doesn't sacrifice reasoning capability

#### DeepSeek-R1 Key Features

**"Aha moment" behavior:**
During RL training, the model spontaneously developed the ability to reconsider:
```
"Wait, I think I made an error. Let me re-examine..."
"Actually, this approach won't work because..."
"Let me try a different method..."
```
This emerged from RL, NOT from supervised data. Pure emergent behavior.

**Language mixing problem:**
During RL training, the model sometimes mixed Chinese and English in reasoning.
Fixed by adding a consistency reward (penalize language mixing).

**Performance:**
- AIME 2024: 79.8% (vs o1: 74.3%, GPT-4: 9.3%)
- Fully open weights + training methodology
- Community immediately created distilled versions (7B-70B)

---

### The Process Reward Model (PRM) — Key Concept

**Outcome Reward:** Correct final answer = +1, wrong = 0
**Process Reward:** Score each reasoning step

```
Problem: "Is 17 prime?"
ORM approach: only reward "yes" or "no"

PRM approach:
Step 1: "I need to check divisibility"        → +0.9 (correct approach)
Step 2: "Check 2: 17/2 = 8.5, not divisible" → +1.0 (correct)
Step 3: "Check 3: 17/3 = 5.67, not divisible"→ +1.0 (correct)
Step 4: "Check 5: 17/5 = 3.4, not divisible" → +1.0 (correct)
Step 5: "No divisor up to √17, so 17 is prime"→ +1.0 (correct)
```

PRMs enable:
- Verification of reasoning (not just final answer)
- Better RL signals (reward correct intermediate steps)
- Beam search over reasoning steps (try multiple paths at each step)

---

## PART 2: MULTIMODAL LLMs

### How Vision-Language Models Work

#### CLIP — Contrastive Language-Image Pretraining (OpenAI, 2021)

The foundation for all modern vision-language models.

**Training:**
```
400M (image, caption) pairs from internet

Image Encoder (ViT) → image embedding
Text Encoder (Transformer) → text embedding

Contrastive loss: maximize similarity of matching pairs,
                 minimize similarity of non-matching pairs

Result: Image and text in the SAME embedding space
```

**Why this matters:**
```
"A cat sitting on a mat" → [0.2, 0.8, ...]   (text vector)
[photo of cat on mat]   → [0.21, 0.79, ...]  (image vector — SIMILAR!)
[photo of dog]          → [0.9, 0.1, ...]    (image vector — DIFFERENT)
```

CLIP unified vision and language in one vector space.
This enables zero-shot image classification, visual search, and is the
visual backbone of almost every multimodal LLM.

#### Vision Transformer (ViT) — How Images Become Tokens

Standard Transformers process sequences of tokens.
Images must be converted to "visual tokens" first.

```
Image: 224 × 224 pixels (RGB)

Step 1: Divide into patches of 16×16 pixels
        Number of patches: (224/16)² = 196 patches

Step 2: Flatten each patch: 16×16×3 = 768 values

Step 3: Linear projection to d_model dimension

Step 4: Add positional embeddings (2D-aware)

Step 5: Process as sequence of 196 "visual tokens"
        through standard transformer
```

The model learns that patch positions correspond to spatial locations.

#### LLaVA — Large Language and Vision Assistant

**Architecture (simple and elegant):**
```
Image → CLIP ViT (frozen) → visual features
      → MLP projection → visual tokens (same dim as text tokens)
      
Text prompt → text tokens

[visual tokens] + [text tokens] → LLaMA (language model) → output
```

**Training (2 stages):**

Stage 1: Train only the MLP projection (freeze everything else)
- Align visual features with language model embedding space
- 595K image-text pairs

Stage 2: Fine-tune MLP + language model (freeze CLIP)
- Instruction following with images
- 158K visual instruction pairs

**Why freeze CLIP?**
CLIP took enormous compute to train. Its visual representations are
already excellent. We just need to project them into the LLM's space.

#### LLaMA 3.2 Vision

Modern approach (cross-attention instead of token concatenation):
```
Text tokens → Self-Attention layers (standard)
                         ↕ Cross-Attention (every 4th layer)
Image tokens → CLIP ViT
```
Cross-attention lets text tokens attend to visual features.
More efficient than concatenating all visual tokens.

#### GPT-4V / GPT-4o Architecture

Not publicly disclosed, but likely:
- Much larger image resolution support
- Multiple image inputs
- Video frame processing
- Unified audio-vision-language model (GPT-4o)

---

## PART 3: SCALING LAWS

### Kaplan Scaling Laws (OpenAI, 2020)

**Key finding:** LLM performance follows power laws with scale.

```
Loss ∝ N^(-α)   (N = number of parameters)
Loss ∝ D^(-β)   (D = dataset size)
Loss ∝ C^(-γ)   (C = compute budget)

Where α, β, γ ≈ 0.076, 0.095, 0.050 (empirically measured)
```

**Prediction:** 10× more parameters → ~43% lower loss. Consistent improvement.

**The data conclusion:** You could train indefinitely on more data with a fixed model size
and keep improving. More data = better model, no saturation observed.

---

### Chinchilla Scaling Laws (DeepMind, 2022)

**The finding that changed LLM training forever.**

Kaplan suggested: compute budget → mostly scale parameters, train on ~20B tokens.
DeepMind found: **this was wrong.** Models were dramatically undertrained.

**Chinchilla optimal:** For a given compute budget C:
```
N_optimal = (C / 6)^0.5    ← parameters
D_optimal = C / (6N)       ← tokens

Result: N ≈ D  (roughly equal params and tokens)
```

**GPT-3 was ~20× undertrained:**
```
GPT-3: 175B parameters, 300B tokens
Chinchilla optimal for same compute: 70B parameters, 1.4T tokens

Chinchilla 70B outperformed GPT-3 175B on most benchmarks.
```

**Impact on practice:**
- LLaMA 1: 65B params, 1.4T tokens (close to Chinchilla optimal for that budget)
- LLaMA 2: 70B params, 2T tokens (slightly past Chinchilla)
- LLaMA 3: 8B params, 15T tokens (MASSIVELY past Chinchilla)

Why train past Chinchilla? **Inference is cheap; training is amortized.**
A model trained on 10T tokens may cost the same to train as one on 1T tokens,
but you serve 1 billion queries. The per-query cost justifies extra training data.

---

### Emergent Abilities

**Paper:** Wei et al., Google, 2022

**The phenomenon:** At certain model sizes, abilities appear **suddenly** — not gradually.

```
Model scale → 7B: 0% accuracy on math   ← doesn't have the skill
Model scale → 13B: 0% accuracy on math  ← still nothing
Model scale → 70B: suddenly 55% on math ← emerged!
```

These include:
- Multi-step arithmetic
- Causal inference
- Word analogy reasoning
- Chain-of-thought reasoning

**Why do emergent abilities occur?**
Hypothesis 1: The model needs enough capacity to internally represent the full algorithm
Hypothesis 2: Benchmark effects — tasks with partial credit would show gradual improvement
Hypothesis 3: Phase transitions in learned representations

**The controversy:** Schaeffer et al. (2023) argued emergent abilities are an artifact of
metrics — switch to smooth metrics and you see smooth improvement.
The debate continues.

**What you say in an interview:**
> "Emergent abilities are capabilities that appear to arise sharply at certain model
> scales, like few-shot reasoning or chain-of-thought. Whether they're truly emergent
> or an artifact of discrete metrics is debated, but the practical reality is that
> many capabilities only become useful at certain scale thresholds."

---

### LLM Benchmarks — What They Actually Test

#### MMLU — Massive Multitask Language Understanding

- **Format:** 4-choice multiple choice questions
- **Coverage:** 57 academic subjects (history, science, law, medicine, math, coding...)
- **Standard eval:** 5-shot (5 examples before the question)
- **What it measures:** Breadth of factual knowledge
- **Top scores:** GPT-4 ~86%, LLaMA 3 70B ~82%, Mistral 7B ~65%
- **Limitation:** Multiple choice may not reflect real-world use

#### HumanEval — Code Generation

- **Format:** 164 Python programming problems with unit tests
- **Metric:** pass@k — probability of generating a correct solution in k tries
- **Standard:** pass@1 (one attempt)
- **What it measures:** Code generation capability
- **Top scores:** GPT-4 ~87%, DeepSeek-Coder-V2 ~85%, LLaMA 3 70B ~72%

#### GSM8K — Grade School Math

- **Format:** 8,500 grade school math word problems
- **What it measures:** Multi-step arithmetic reasoning
- **Telling score:** Where CoT dramatically helps (5% → 50%+ with CoT on GPT-3)

#### MATH — Competition Math

- **Format:** 12,500 high school competition math problems (AMC, AIME level)
- **Very hard:** GPT-4 ~52%, o1 ~94%

#### MT-Bench — Multi-Turn Conversation

- **Format:** 80 high-quality multi-turn conversations across 8 categories
- **Metric:** GPT-4 rates responses 1-10
- **What it measures:** Instruction following, conversation quality

#### Chatbot Arena — Human Preference Elo

- **Format:** Real users compare two models blindly, vote for better
- **Metric:** Elo rating (same as chess)
- **Gold standard for:** Real-world user preference
- **As of 2025:** GPT-4o, Claude 3.5, Gemini 1.5 Pro near the top

#### GPQA — Graduate-Level Questions

- **Format:** PhD-level science questions (biology, chemistry, physics)
- **Even PhDs only score ~65% on their own domain**
- **What it measures:** Deep scientific reasoning
- **o1:** 78%, GPT-4: 53%

---

## INTERVIEW BLAST

**"What is o1 and how was it trained?"**
> "o1 is OpenAI's reasoning model trained with RL using process reward models.
> Unlike standard LLMs that predict one token at a time, o1 spends compute at
> inference time doing extended chain-of-thought reasoning before answering.
> The PRM scores each reasoning step (not just the final answer), enabling
> the model to learn to backtrack, verify, and explore multiple approaches.
> It scores 74% on AIME math competition questions versus GPT-4's 13%."

**"What is DeepSeek-R1 and what is GRPO?"**
> "DeepSeek-R1 is the first open-source reasoning model matching o1. GRPO
> (Group Relative Policy Optimization) is the RL algorithm used — it samples
> multiple responses to each problem, uses the group average reward as baseline,
> and updates the policy to favor above-average responses. No separate critic
> network needed (unlike PPO), making it more memory-efficient."

**"Explain Chinchilla scaling laws."**
> "Chinchilla showed that compute should be split equally between model size and
> training data. GPT-3 trained 175B parameters on 300B tokens — Chinchilla showed
> 70B parameters on 1.4T tokens achieves better performance for the same compute.
> The key insight: models were dramatically undertrained. LLaMA 3 takes this further,
> training 8B parameters on 15T tokens — far past Chinchilla optimal — because
> inference cost justifies extra training."
