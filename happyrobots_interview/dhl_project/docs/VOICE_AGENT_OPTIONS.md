# Best Voice Agent for the DHL Demo — Honest Comparison

The goal: a voice agent that answers a phone call, understands speech, and calls your **n8n workflow** as its brain, then speaks the reply back. So you need: telephony + Speech-to-Text (STT) + Text-to-Speech (TTS), with a webhook to n8n.

---

## 🏆 Recommendation for YOUR demo: **Vapi** (or Retell AI)

For a fresher building a showcase that plugs into n8n, **Vapi** is the best balance of: fast setup, free trial credits, great voice quality, and a clean webhook to your workflow. **Retell AI** is an equally good alternative — pick whichever UI you like.

---

## The Comparison

| Platform | What it is | Best for | Free tier | Connects to n8n? |
|----------|-----------|----------|-----------|------------------|
| **Vapi** ⭐ | Full voice-agent platform (telephony + STT + TTS) | Fast, great-sounding demos | Trial credits (~$10) | ✅ Custom LLM / webhook → your n8n |
| **Retell AI** ⭐ | Voice-agent platform, very low latency | Natural back-and-forth calls | Trial credits | ✅ Webhook / custom LLM |
| **ElevenLabs** | Best-in-class TTS + Conversational AI agents | When the VOICE must sound stunning | Free chars + free agent mins | ✅ Webhook tools |
| **Twilio** | Telephony backbone (you add STT/TTS) | The "enterprise" answer | ~$15 trial credit | ✅ Already wired in your workflow |
| **OpenAI Realtime API** | Speech-to-speech, ultra low latency | Cutting-edge, you build more | Pay-as-you-go | ⚙️ More custom code |
| **Synthflow** | No-code voice agent builder | Non-developers | Limited trial | ✅ Webhook |

---

## Why Vapi/Retell over the others (for you, right now)

- **Setup in under an hour** — you create an agent, give it a phone number, point it at a webhook. No telephony plumbing.
- **The voice sounds genuinely good** — they use ElevenLabs/Deepgram/Azure voices under the hood.
- **Low latency** — built for real conversations, not robotic pauses.
- **Free credits** — enough to record your demo without spending money (you're a fresher, this matters).
- **It calls YOUR n8n workflow** — so your 5-agent system stays the brain. The voice platform is just the ears and mouth.

**Twilio** is the most "enterprise correct" answer (it's what a real DHL would likely use as the backbone), so **mention it in the interview** — but it's more work to wire than Vapi for a quick demo.

---

## How It Connects to Your Workflow (the architecture)

```
  Customer calls the Vapi/Retell phone number
            │
            ▼
  Vapi handles: pick up → Speech-to-Text (STT)
            │
            ▼   POST { message, channel:"voice", from }
  ──────►  Your n8n  /webhook/dhl-omnichannel
            │        (Normalize → Router → Agent → reply text)
            ▼   returns the reply text
  Vapi handles: Text-to-Speech (TTS) → speaks it to the caller
```

Your workflow is **already built for this** — the Normalize node accepts `{ message, channel, from }`, and every agent already shortens its reply when `channel = voice`. You just point Vapi's "server URL / custom LLM webhook" at your n8n webhook.

> ⚠️ One mapping note: configure Vapi to POST the transcribed text as `message` and set `channel` to `"voice"`. The Normalize node does the rest.

---

## Voice Quality Tips (so it doesn't sound robotic)

- Pick an **Indian English voice** (Vapi/ElevenLabs both have them) so DHL India names and cities sound right.
- Keep replies **1-2 short sentences** — your agents already do this for voice.
- Add a short **filler while it thinks**: "Let me check that for you…" so there's no dead silence.
- Spell numbers naturally — "eight PM," not "8 PM" — your voice prompt already instructs this.

---

## What To Say In The Interview

> "For voice I'd use a platform like Vapi or Retell that handles telephony, speech-to-text and text-to-speech, and calls my n8n workflow as the brain via webhook — so the five-agent intelligence stays in one place and the voice layer is swappable. For a real DHL deployment the backbone would more likely be Twilio with Azure or Deepgram for STT and ElevenLabs or Azure Neural for TTS, wrapped in SSML for tone and clear number pronunciation. The key design choice is keeping the agent logic channel-agnostic so voice, SMS, email and chat all reuse the same brain."

That answer shows you know the **landscape**, the **architecture**, and the **enterprise reality** — senior-level thinking from a fresher.

---

*Pricing and free tiers change — verify current rates on each platform's site before recording your demo.*
