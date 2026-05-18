# Reasoning Models, Multimodal LLMs & Scaling Laws

> o1, DeepSeek-R1, vision LLMs, and why bigger = better (until it doesn't).
> Critical topics missing from every basic GenAI curriculum.

---

## PART 1: REASONING MODELS — O1, O3, DEEPSEEK-R1

### Why Standard LLMs Fail at Reasoning

**What it is:** The fundamental limitation of next-token prediction for complex reasoning — fast, fluent text generation is not the same as careful thinking.

Standard LLMs predict the next token. Fast, fluent, but:
```
Q: "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball.
    How much does the ball cost?"

GPT-3.5 (fast answer): "$0.10"   ← WRONG (common cognitive bias)
Correct answer: $0.05 (bat=$1.05, ball=$0.05, difference=$1.00 ✓)
```

**WHY this happens:** GPT-3.5 sees "$1.10 total" and "$1.00 more" and pattern-matches to "$0.10" — the most common response to this type of question in its training data. It doesn't actually solve the algebra. It's predicting what a correct-sounding answer looks like, not computing the answer.

The model gives the intuitive wrong answer because it was trained to be fast,
not to reason carefully.

**Chain-of-Thought helps but has limits.** You can prompt "think step by step"
but the model wasn't trained to explore multiple reasoning paths, backtrack,
or verify its own work.

**Analogy:** A student who memorizes common exam questions vs a student who learns to actually solve problems. The memorizer does great on familiar questions but fails on novel ones. Reasoning models try to create the second type of student.

---

### OpenAI o1 — Process Reward + RL for Reasoning

**What it is:** OpenAI's model that spends significant computation at inference time "thinking through" problems before answering — trained with reinforcement learning to reward good reasoning steps, not just correct final answers.

**Released:** September 2024
**Core idea:** Train the model to spend compute at **inference time** doing extended reasoning before answering.

#### How o1 Was Trained (What We Know)

**Step 1: Collect long reasoning chains**

**What it is:** Building a dataset of correct multi-step reasoning before training begins.

Generate many long chains-of-thought (CoT) for hard problems (math, coding, logic).
Use human experts to label which reasoning steps are **correct** vs **incorrect**.

**Step 2: Train a Process Reward Model (PRM)**

**What it is:** A separate neural network that scores the quality of each reasoning step — not just the final answer.

```
Standard reward model (Outcome Reward Model): scores the FINAL answer (correct/incorrect)
  "The answer is $0.05" → score 1.0 (correct)
  "The answer is $0.10" → score 0.0 (wrong)
  ← Only rewards the destination, not the journey

Process reward model (PRM): scores each STEP in the reasoning chain
  PRM("Step: Let x be the ball price. Then bat = x + 1.00...") → 0.9 (good step)
  PRM("Step: The ball costs $0.10 because 1.10-1.00=0.10...")  → 0.1 (bad step)
  ← Rewards each waypoint on the journey to the answer
```

**WHY process reward over outcome reward?** With only outcome rewards, a model learns "get the right answer somehow." With process rewards, it learns "reason correctly at each step." A model with correct reasoning will generalize to new problems; a model that learned to guess correctly on training problems won't.

This is the key innovation: reward **correct reasoning steps**, not just correct answers.

**Step 3: RL with the PRM as reward**

**What it is:** Using reinforcement learning where the PRM scores each reasoning step — training the model to generate chains of thought where every step is sound.

Use reinforcement learning where each reasoning step gets a reward from the PRM.
The model learns to:
- Generate correct intermediate steps (not just plausible-sounding ones)
- Backtrack when a step seems wrong (catch its own mistakes)
- Try multiple approaches (like a human thinking hard through alternatives)
- Verify its answer before committing (check the work)

**Step 4: Inference = extended thinking**

**What it is:** At test time, o1 generates a long hidden thinking process before outputting an answer — spending tokens on reasoning rather than on the answer text.

At test time, o1 generates a long "thinking" process (hidden from user) before answering.
More thinking tokens = better accuracy on hard problems.

```
User: "Solve this integral: ∫ x²e^x dx"

o1 internal thinking (hidden from user — shows in "thinking" block):
  "Let me try integration by parts. u = x², dv = e^x dx
   Then du = 2x dx, v = e^x
   ∫ x²e^x dx = x²e^x - ∫ 2x e^x dx
   Now I need ∫ 2x e^x dx, use parts again:
     u = 2x, dv = e^x dx
     du = 2dx, v = e^x
     ∫ 2x e^x dx = 2x e^x - ∫ 2e^x dx = 2x e^x - 2e^x
   Back to main integral:
   ∫ x²e^x dx = x²e^x - (2x e^x - 2e^x) + C
               = x²e^x - 2x e^x + 2e^x + C
               = (x²-2x+2)e^x + C
   Let me verify by differentiating: d/dx[(x²-2x+2)e^x]
     = (2x-2)e^x + (x²-2x+2)e^x = (x²)e^x ✓"
← This entire thinking process costs many tokens but produces a verified answer

o1 output (shown to user): "(x²-2x+2)e^x + C"
← Short, clean, correct answer after the thinking
```

#### o1 Performance

**What it is:** Benchmark comparisons that show the dramatic improvement reasoning models achieve over standard LLMs on hard problems.

- AIME (American Invitational Mathematics Exam): 74% (vs GPT-4: 13%)
- Competition math (MATH benchmark): 94% (vs GPT-4: 72%)
- PhD-level science questions (GPQA): 78% (vs GPT-4: 53%)

The cost: o1 uses 10-100× more tokens per query than GPT-4.
Slower, more expensive, but dramatically better on hard reasoning tasks.

**WHY the tradeoff is worth it:** A human expert consultant charges $500/hour. If o1 solving a $10,000 math problem costs $2 in tokens vs $0.20 for GPT-4, the accuracy improvement is worth the 10× cost increase. You pay for reasoning when the stakes are high.

---

### DeepSeek-R1 — Fully Open Reasoning Model

**What it is:** The first fully open-source reasoning model that matched o1's performance — AND published the training methodology transparently, enabling the whole community to reproduce it.

**Released:** January 2025
**Significance:** First open-source reasoning model matching o1. Fully transparent training.

#### The Training Pipeline (4 Stages)

**Stage 1: Cold Start with Long CoT Data**

**What it is:** Bootstrapping the RL training by first teaching the model the FORMAT of long reasoning chains.

```
Problem: RL from scratch produces incoherent reasoning
← The model doesn't know how to structure a reasoning chain
← It might generate random thoughts, not systematic problem solving

Solution: First collect or generate ~thousands of long CoT examples
          Fine-tune the base model on these (SFT)
          Now the model knows how to format long reasoning
          ← Like teaching a student how to show their work before testing them
```

**Stage 2: GRPO Training (Reinforcement Learning)**

**What it is:** The core RL algorithm used — a PPO variant that doesn't need a separate critic network, making it more memory-efficient and stable.

GRPO = Group Relative Policy Optimization

Instead of PPO (which needs a separate value/critic network), GRPO:
1. For each problem, sample G responses from the current policy
2. Compute reward for each response (rule-based: is the answer correct?)
3. Use the group's average reward as the baseline
4. Update policy to increase probability of above-average responses

```python
# GRPO Training Loop (conceptual):
def grpo_update(problem, policy_model, G=8):
    # Step 1: Sample G responses to the same problem
    responses = [policy_model.generate(problem) for _ in range(G)]
    # responses: 8 different reasoning chains and answers
    # Each response uses the CURRENT model weights (same checkpoint)

    # Step 2: Compute rewards for each response
    rewards = []
    for response in responses:
        answer = extract_final_answer(response)  # parse the answer from the chain
        is_correct = verify_answer(answer, ground_truth)  # check correctness
        reward = 1.0 if is_correct else 0.0  # simple binary reward
        # WHY binary? For math/coding, the answer is right or wrong — no partial credit needed
        rewards.append(reward)

    # Step 3: Compute group baseline (average reward)
    baseline = sum(rewards) / G  # e.g., 0.625 if 5 of 8 are correct

    # Step 4: Compute advantages (how much better than average?)
    advantages = [r - baseline for r in rewards]
    # Correct responses: advantage = 1.0 - 0.625 = +0.375 (above average, reinforce)
    # Wrong responses: advantage = 0.0 - 0.625 = -0.625 (below average, discourage)

    # Step 5: Update policy
    # Increase probability of high-advantage responses
    # Decrease probability of low-advantage responses
    policy_loss = -mean(advantages * log_probs_of_responses)
    # ← This is the GRPO objective: push toward above-average, away from below-average
    policy_model.backward(policy_loss)

# Example:
"""
G=8 responses to "What is 15% of 340?"
  Response 1: "Let me calculate 15/100 × 340 = 0.15 × 340 = 51" → answer "51" → reward = 1
  Response 2: "15% = 15/100. 340 × 15 / 100 = 5100/100 = 51" → "51" → reward = 1
  Response 3: "10% of 340 = 34, 5% = 17, total 15% = 51" → "51" → reward = 1
  Response 4: "15 × 340 = 5100, so 15% = 5100/100 = 51" → "51" → reward = 1
  Response 5: "15% ≈ 0.15, 0.15 × 340 ≈ 52" → "52" → reward = 0  (calculation error)
  Response 6: "15% is 51" → "51" → reward = 1
  Response 7: "340 / 15 = 22.7" → "22.7" → reward = 0  (wrong operation)
  Response 8: "15/340 = 0.044" → "0.044" → reward = 0  (inverted)
  
  Group average = 5/8 = 0.625
  Responses 1-4,6 get positive advantage (+0.375) — reinforce these patterns
  Responses 5,7,8 get negative advantage (-0.625) — discourage these patterns
"""
```

**Why GRPO over PPO?**
- **No separate critic network** (saves memory — PPO needs 2× model parameters for critic)
- **More stable training** — simpler algorithm
- **Rule-based rewards** (no reward model to train separately)
- **Works well for verifiable domains** (math, coding — clear right/wrong)

**Stage 3: Rejection Sampling Fine-tuning**

**What it is:** Using the RL-trained model to generate many responses, keeping only the high-quality ones, then fine-tuning again — a data quality boost.

- Generate many responses using the RL-trained model
- Keep only high-quality responses (correct AND clear reasoning)
- Fine-tune the model on these filtered high-quality examples

**WHY this helps:** The RL model is now better than the original base model. Its best outputs represent high-quality reasoning chains we can learn from. Filtering out bad responses and training on the good ones raises the floor — the model becomes consistently good, not just occasionally good.

**Stage 4: Final RLHF (for helpfulness + safety)**
- Standard DPO/RLHF to make the model helpful and safe
- Doesn't sacrifice reasoning capability

#### DeepSeek-R1 Key Features

**"Aha moment" behavior:**

**What it is:** An emergent behavior that appeared during RL training without being explicitly programmed — the model spontaneously developed the ability to reconsider its own reasoning.

During RL training, the model spontaneously developed the ability to reconsider:
```
"Wait, I think I made an error. Let me re-examine..."
← The model notices inconsistency in its own reasoning

"Actually, this approach won't work because..."
← The model reasons about WHY its current approach is flawed

"Let me try a different method..."
← The model pivots to an alternative approach
```
This emerged from RL, NOT from supervised data. Pure emergent behavior.

**WHY this is remarkable:** Nobody programmed "detect your own mistakes and backtrack." The model discovered this strategy on its own because it led to better rewards. It's analogous to a student who, through trial and error on tests, discovers that checking their work improves their score.

**Language mixing problem:**

**What it is:** An unintended behavior that emerged during training — and how it was fixed.

During RL training, the model sometimes mixed Chinese and English in reasoning.
(DeepSeek is a Chinese lab; the base model saw lots of Chinese text)
Fixed by adding a consistency reward (penalize language mixing in reasoning chains).

**Performance:**
- AIME 2024: 79.8% (vs o1: 74.3%, GPT-4: 9.3%)
- Fully open weights + training methodology
- Community immediately created distilled versions (7B-70B with reasoning capability)

---

### The Process Reward Model (PRM) — Key Concept

**What it is:** A model that evaluates the quality of each reasoning step individually — enabling richer training signal and step-level verification of reasoning chains.

**Outcome Reward:** Correct final answer = +1, wrong = 0
**Process Reward:** Score each reasoning step individually

```
Problem: "Is 17 prime?"
ORM approach: only reward "yes" or "no" (outcome)
← Binary: doesn't tell the model WHICH steps to improve

PRM approach:
Step 1: "I need to check divisibility"        → PRM score: +0.9 (correct approach — trial division)
Step 2: "Check 2: 17/2 = 8.5, not divisible" → PRM score: +1.0 (correct calculation)
Step 3: "Check 3: 17/3 = 5.67, not divisible"→ PRM score: +1.0 (correct)
Step 4: "Check 5: 17/5 = 3.4, not divisible" → PRM score: +1.0 (correct)
Step 5: "No divisor up to √17, so 17 is prime"→ PRM score: +1.0 (correct conclusion with proof)
← Each step independently validated — model gets signal on WHERE it went wrong
```

PRMs enable:
- **Verification of reasoning** (not just final answer) — trust the work, not just the result
- **Better RL signals** (reward correct intermediate steps) — more informative training
- **Beam search over reasoning steps** (try multiple paths at each step) — at test time, explore alternatives

**WHY PRM is more powerful for training:** With outcome reward, if the model gets a wrong final answer, it doesn't know WHICH of its 20 reasoning steps was wrong. With PRM, it gets step-by-step feedback — like a math teacher who marks each line of your work, not just the final answer.

---

## PART 2: MULTIMODAL LLMs

### How Vision-Language Models Work

**What it is:** The architecture and training approach for models that understand both images and text — the foundation for every visual AI system.

#### CLIP — Contrastive Language-Image Pretraining (OpenAI, 2021)

**What it is:** A model trained to understand that images and their text descriptions "mean the same thing" — creating a shared embedding space where visual and textual concepts align.

The foundation for all modern vision-language models.

**Training:**
```
400M (image, caption) pairs from internet
← Massive scale of naturally paired image-text data

Image Encoder (ViT) → image embedding vector
Text Encoder (Transformer) → text embedding vector

Contrastive loss:
  Matching pairs (image, caption): push embeddings CLOSE together (high similarity)
  Non-matching pairs (image, random caption): push embeddings FAR apart (low similarity)

Result: Image and text in the SAME embedding space
        ← "visual bank" (photo) and "text bank" (word) map to similar vectors
```

```python
# How CLIP works conceptually:
import clip

model, preprocess = clip.load("ViT-B/32")

# Encode an image:
image = preprocess(Image.open("cat.jpg")).unsqueeze(0)
image_features = model.encode_image(image)  # shape: (1, 512)
# ← 512-dimensional vector representing the image's visual content

# Encode text descriptions:
texts = clip.tokenize(["a cat", "a dog", "a car"])
text_features = model.encode_text(texts)  # shape: (3, 512)
# ← 512-dimensional vectors for each text description

# Compute similarity:
similarity = (image_features @ text_features.T)  # shape: (1, 3)
# similarity[0][0] = high (image of cat, text "a cat")
# similarity[0][1] = medium (cat and dog are similar animals)
# similarity[0][2] = low (cat and car are very different)

# Zero-shot image classification:
# The class with highest similarity = predicted label
# No task-specific training needed!
```

**Why this matters:**
```
"A cat sitting on a mat" → [0.2, 0.8, ...]   (text vector)
[photo of cat on mat]   → [0.21, 0.79, ...]  (image vector — SIMILAR! near-identical)
[photo of dog]          → [0.9, 0.1, ...]    (image vector — DIFFERENT)
```

CLIP unified vision and language in one vector space.
This enables zero-shot image classification, visual search, and is the
visual backbone of almost every multimodal LLM.

**WHY contrastive learning works:** The model is forced to find shared representations that explain both modalities. "cat on mat" text and photo of cat on mat must map to similar vectors — the only way to do this is to learn a concept of "cat" and "mat" that's media-agnostic.

#### Vision Transformer (ViT) — How Images Become Tokens

**What it is:** The technique for converting an image into a sequence of "visual tokens" that a Transformer can process — treating image patches the same way tokens treat words.

Standard Transformers process sequences of tokens.
Images must be converted to "visual tokens" first.

```
Image: 224 × 224 pixels (RGB)
← 3 color channels, 224×224 spatial resolution = 150,528 values total

Step 1: Divide into patches of 16×16 pixels
        Number of patches: (224/16)² = 14×14 = 196 patches
        ← Like dividing a photo into a 14×14 grid of small squares

Step 2: Flatten each patch: 16×16×3 = 768 values
        ← Each small square becomes a vector of 768 numbers

Step 3: Linear projection to d_model dimension
        ← Map each 768-value patch to whatever size the model uses (e.g., 512)

Step 4: Add positional embeddings (2D-aware)
        ← Tell the model "this patch is in row 3, column 7"

Step 5: Process as sequence of 196 "visual tokens"
        through standard transformer (same architecture as text!)
        ← The Transformer doesn't know if it's processing text or image patches
```

The model learns that patch positions correspond to spatial locations.
After training: patches from the same image region attend to each other.

```python
# ViT processing an image (simplified):
class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, d_model=768):
        self.patch_embed = nn.Conv2d(
            in_channels=3,           # RGB input
            out_channels=d_model,    # project to model dimension
            kernel_size=patch_size,  # each patch = 16×16 pixels
            stride=patch_size        # no overlap between patches
        )
        # This conv layer simultaneously: splits into patches + projects to d_model

        num_patches = (image_size // patch_size) ** 2  # 196 patches
        # One learned position embedding per patch position (2D grid)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, d_model))

    def forward(self, image):
        # image: (batch, 3, 224, 224) — RGB image
        patches = self.patch_embed(image)
        # patches shape: (batch, d_model, 14, 14) — 14×14 grid of d_model-dim vectors
        patches = patches.flatten(2).transpose(1, 2)
        # Flatten spatial: (batch, 196, d_model) — 196 "visual tokens"
        tokens = patches + self.pos_embed
        # Add position info: each patch knows where it is in the image grid
        return self.transformer(tokens)  # process as sequence of tokens
```

#### LLaVA — Large Language and Vision Assistant

**What it is:** An elegant, simple architecture that connects a visual encoder (CLIP) to a language model (LLaMA) with a small learned projection layer.

**Architecture (simple and elegant):**
```
Image → CLIP ViT (frozen) → visual features (196 vectors of dim 1024)
      → MLP projection → visual tokens (same dim as text tokens, e.g. 4096)
      ← The MLP "translates" visual features into the LLM's language

Text prompt → text tokens (same shape: seq_len × 4096)

[visual tokens] + [text tokens] → LLaMA (language model) → output
← Concatenate image tokens and text tokens into one sequence
← LLaMA processes them all together with self-attention
← The language model can "read" the image through the projected tokens
```

**Training (2 stages):**

Stage 1: Train only the MLP projection (freeze everything else)
- Align visual features with language model embedding space
- 595K image-text pairs
- Goal: make CLIP's visual features understandable to LLaMA

Stage 2: Fine-tune MLP + language model (freeze CLIP)
- Instruction following with images
- 158K visual instruction pairs (question+image → answer)

```python
# LLaVA forward pass:
class LLaVA(nn.Module):
    def __init__(self):
        self.vision_encoder = CLIPVisualEncoder(frozen=True)
        # CLIP's visual encoder — already trained on 400M image-text pairs
        # Frozen because CLIP's representations are already excellent

        self.projection = nn.Sequential(
            nn.Linear(1024, 4096),  # CLIP dim → LLaMA dim
            nn.GELU(),              # non-linearity
            nn.Linear(4096, 4096)   # refine projection
        )
        # This small MLP is the "translation layer" between visual and text space
        # Trained in Stage 1 to align the two embedding spaces

        self.language_model = LLaMA(frozen=False)
        # Fine-tuned in Stage 2 to follow visual instructions

    def forward(self, image, text_tokens):
        # Step 1: Extract visual features from image
        visual_features = self.vision_encoder(image)  # (batch, 196, 1024)
        # CLIP gives us 196 patch representations, each 1024-dimensional

        # Step 2: Project to language model dimension
        visual_tokens = self.projection(visual_features)  # (batch, 196, 4096)
        # Now visual tokens are in the same 4096-dimensional space as text tokens
        # The language model can "read" them

        # Step 3: Concatenate visual + text tokens
        combined = torch.cat([visual_tokens, text_tokens], dim=1)
        # shape: (batch, 196 + text_len, 4096)
        # Image comes first, then text — model reads image then text

        # Step 4: Language model processes everything together
        output = self.language_model(combined)
        return output
        # The LLM's self-attention can now cross-reference text tokens with visual tokens
        # "What color is the cat?" → text attends to visual tokens showing cat color
```

**Why freeze CLIP?**
CLIP took enormous compute to train. Its visual representations are
already excellent. We just need to project them into the LLM's space.

#### LLaMA 3.2 Vision

**What it is:** A more sophisticated approach to multimodal — instead of token concatenation, uses cross-attention between text and visual layers.

Modern approach (cross-attention instead of token concatenation):
```
Text tokens → Self-Attention layers (standard) → attend to each other normally
                         ↕ Cross-Attention (every 4th layer)
Image tokens → CLIP ViT → visual features (kept separate from text sequence)
```
Cross-attention lets text tokens attend to visual features without mixing them.
More efficient than concatenating all visual tokens into the text sequence.

**WHY cross-attention is more efficient:** Concatenating 196 visual tokens to every text sequence costs 196 extra tokens of attention computation for every layer. Cross-attention only computes vision-text interaction at select layers — cheaper overall.

#### GPT-4V / GPT-4o Architecture

Not publicly disclosed, but likely:
- Much larger image resolution support (can read text in images, charts)
- Multiple image inputs in one conversation
- Video frame processing (GPT-4o)
- Unified audio-vision-language model (GPT-4o handles audio natively)

---

## PART 3: SCALING LAWS

### Kaplan Scaling Laws (OpenAI, 2020)

**What it is:** The empirical discovery that LLM performance follows precise mathematical laws as you increase model size, data, or compute — making the field predictable and investable.

**Analogy:** Like discovering that a factory's output scales exactly with the square root of its investment. You can now plan with certainty: "if I spend 4× more, I get 2× more output." Before scaling laws, AI felt unpredictable. After, it became engineering.

**Key finding:** LLM performance follows power laws with scale.

```
Loss ∝ N^(-α)   (N = number of parameters)
← 10× more parameters → loss decreases by factor 10^0.076 ≈ 1.4× better

Loss ∝ D^(-β)   (D = dataset size in tokens)
← 10× more training data → loss decreases by factor 10^0.095 ≈ 1.5× better

Loss ∝ C^(-γ)   (C = compute budget in FLOPS)
← 10× more compute → loss decreases by factor 10^0.050 ≈ 1.12× better

Where α, β, γ ≈ 0.076, 0.095, 0.050 (empirically measured by OpenAI)
← These exponents are remarkably stable across many orders of magnitude
```

**Prediction:** 10× more parameters → ~43% lower loss. Consistent improvement.

**The data conclusion:** You could train indefinitely on more data with a fixed model size
and keep improving. More data = better model, no saturation observed.

**WHY this was revolutionary:** Before scaling laws, people thought bigger models would hit a wall. If there's a ceiling, why invest? Scaling laws showed no ceiling — a reliable roadmap. This justified billions in investment and predicted GPT-3 would be useful before it was built.

---

### Chinchilla Scaling Laws (DeepMind, 2022)

**What it is:** A correction to Kaplan's laws that showed most models were dramatically undertrained — the optimal ratio of data to parameters is much higher than previously thought.

**The finding that changed LLM training forever.**

Kaplan suggested: compute budget → mostly scale parameters, train on ~20B tokens.
DeepMind found: **this was wrong.** Models were dramatically undertrained.

**Chinchilla optimal:** For a given compute budget C:
```
N_optimal = (C / 6)^0.5    ← optimal number of parameters
D_optimal = C / (6N)       ← optimal number of training tokens

Result: N ≈ D  (roughly equal params and tokens — 1 token per parameter)
← For 70B params, train on ~1.4 trillion tokens (not 300 billion!)
```

**GPT-3 was ~20× undertrained:**
```
GPT-3: 175B parameters, 300B tokens
       ← 300B tokens for 175B params: only 1.7 tokens per parameter
       
Chinchilla optimal for same compute: 70B parameters, 1.4T tokens
       ← 1.4T tokens for 70B params: 20 tokens per parameter

Chinchilla 70B outperformed GPT-3 175B on most benchmarks.
← 2.5× fewer parameters, trained on 4.7× more data — wins!
```

**Impact on practice:**
```
LLaMA 1: 65B params, 1.4T tokens  ← close to Chinchilla optimal for that compute budget
LLaMA 2: 70B params, 2T tokens    ← slightly past Chinchilla optimal
LLaMA 3: 8B params, 15T tokens    ← MASSIVELY past Chinchilla (~1875 tokens/param!)
```

Why train past Chinchilla? **Inference is cheap; training is amortized.**

```python
# The economic argument for over-training:
# Scenario: build a model to serve 1 billion queries

# Option A: Chinchilla-optimal 70B model
training_cost_A = compute_for_chinchilla_optimal(70e9)   # some large number
inference_cost_per_query_A = 70e9 * tokens_per_query    # proportional to params
total_cost_A = training_cost_A + 1e9 * inference_cost_per_query_A
# ← inference cost dominates at 1B queries

# Option B: Over-train 8B model to match 70B quality
training_cost_B = compute_for_llama3(8e9, 15e12)       # higher training cost
inference_cost_per_query_B = 8e9 * tokens_per_query     # 8.75× cheaper per query!
total_cost_B = training_cost_B + 1e9 * inference_cost_per_query_B
# ← much lower inference cost at scale

# If 1B queries: inference savings justify 10× higher training cost
# LLaMA 3's strategy: spend more on training once, save 10× on every query forever
```

---

### Emergent Abilities

**What it is:** Capabilities that appear suddenly at certain model scales — not gradually — which was a surprise to researchers who expected smooth improvement.

**Paper:** Wei et al., Google, 2022

**The phenomenon:** At certain model sizes, abilities appear **suddenly** — not gradually.

```
Model scale → 7B:  0% accuracy on math   ← doesn't have the skill at all
Model scale → 13B: 0% accuracy on math   ← still nothing, no improvement
Model scale → 70B: suddenly 55% on math  ← emerged! jumps from 0% to 55%
```

**Analogy:** Like learning a language. You study for months and barely understand anything. Then suddenly, after enough vocabulary, you start understanding whole sentences. There's a threshold of understanding — below it you have nothing, above it you have competency. Emergent abilities seem similar.

These include:
- Multi-step arithmetic
- Causal inference
- Word analogy reasoning
- Chain-of-thought reasoning

**Why do emergent abilities occur?**
Hypothesis 1: The model needs enough capacity to internally represent the full algorithm
(e.g., to do multi-step math, you need to track intermediate values — requires enough neurons)

Hypothesis 2: Benchmark effects — tasks with partial credit would show gradual improvement
(binary benchmarks appear discontinuous even when underlying capability grows smoothly)

Hypothesis 3: Phase transitions in learned representations
(like water freezing — gradual cooling, then suddenly it's solid at exactly 0°C)

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

**What it is:** The standard evaluation datasets used to compare models — knowing what each tests (and its limitations) shows technical sophistication.

#### MMLU — Massive Multitask Language Understanding

**What it is:** The "SAT for LLMs" — a broad multiple-choice test across 57 academic subjects.

- **Format:** 4-choice multiple choice questions
- **Coverage:** 57 academic subjects (history, science, law, medicine, math, coding...)
- **Standard eval:** 5-shot (5 examples shown before each test question)
- **What it measures:** Breadth of factual knowledge
- **Top scores:** GPT-4 ~86%, LLaMA 3 70B ~82%, Mistral 7B ~65%
- **Limitation:** Multiple choice may not reflect real-world use (you can get 25% by guessing)

#### HumanEval — Code Generation

**What it is:** A coding challenge benchmark where models write Python functions, tested by running them against unit tests.

- **Format:** 164 Python programming problems with unit tests
- **Metric:** pass@k — probability of generating a correct solution in k tries
- **Standard:** pass@1 (one attempt)
- **What it measures:** Code generation capability
- **Top scores:** GPT-4 ~87%, DeepSeek-Coder-V2 ~85%, LLaMA 3 70B ~72%

#### GSM8K — Grade School Math

**What it is:** Elementary word problems requiring multi-step arithmetic — tests basic mathematical reasoning, not advanced math.

- **Format:** 8,500 grade school math word problems
- **What it measures:** Multi-step arithmetic reasoning
- **Telling score:** Where CoT dramatically helps (5% → 50%+ with CoT on GPT-3)

#### MATH — Competition Math

**What it is:** High school competition math (AMC/AIME level) — genuinely difficult problems requiring real mathematical reasoning.

- **Format:** 12,500 high school competition math problems (AMC, AIME level)
- **Very hard:** GPT-4 ~52%, o1 ~94% (the gap shows what reasoning models achieve)

#### MT-Bench — Multi-Turn Conversation

**What it is:** A benchmark testing whether models can maintain helpful, accurate multi-turn conversations — closer to real use than single-question benchmarks.

- **Format:** 80 high-quality multi-turn conversations across 8 categories
- **Metric:** GPT-4 rates responses 1-10
- **What it measures:** Instruction following, conversation quality

#### Chatbot Arena — Human Preference Elo

**What it is:** Real users comparing two anonymous models and voting — the closest thing to measuring actual user satisfaction.

- **Format:** Real users compare two models blindly, vote for better
- **Metric:** Elo rating (same as chess — win/loss against opponents of known strength)
- **Gold standard for:** Real-world user preference
- **As of 2025:** GPT-4o, Claude 3.5, Gemini 1.5 Pro near the top

**WHY Chatbot Arena matters most:** It's the only benchmark where real users with real tasks are evaluating. All other benchmarks can be gamed by training on benchmark-adjacent data. Chatbot Arena reflects what people actually want.

#### GPQA — Graduate-Level Questions

**What it is:** PhD-level science questions so hard that even domain experts frequently get them wrong — designed to test deep reasoning, not memorization.

- **Format:** PhD-level science questions (biology, chemistry, physics)
- **Even PhDs only score ~65% on their own domain** — these are genuinely hard
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

**"What are emergent abilities?"**
> "Capabilities that appear suddenly at certain model scales — like multi-step
> arithmetic that doesn't exist at 7B or 13B but suddenly emerges at 70B.
> Whether these are truly discontinuous jumps or artifacts of binary benchmarks
> is debated. Practically, many capabilities only become useful above certain
> scale thresholds — which is why larger models unlock new application types."

**"How do vision-language models work?"**
> "Modern VLMs like LLaVA use a CLIP encoder to convert images into 196 visual
> tokens (one per 16×16 patch), a small MLP projection to align those tokens
> with the language model's embedding space, then a standard language model
> processes the concatenated visual and text tokens. CLIP is frozen (already trained
> on 400M image-text pairs). Only the projection MLP and language model fine-tune.
> More modern approaches like LLaMA 3.2 use cross-attention instead of concatenation
> for efficiency."
