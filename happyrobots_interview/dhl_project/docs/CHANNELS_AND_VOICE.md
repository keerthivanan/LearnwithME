# Omnichannel & Voice — How One Workflow Serves Voice, SMS, Email & Chat

This is the JD line made real:
> "Configure multi-channel AI experiences across **voice, email, SMS, and chat**."

The DHL AI Command Center is built so **one** workflow handles **all four channels** — the agents and tools never change, only the way the message arrives and the way the reply is delivered.

---

## The Core Idea — Normalize Once, Serve Everywhere

```
  💬 Chat widget ─────────┐
  📞 Twilio Voice ────────┤
  📱 Twilio SMS/WhatsApp ─┼──►  🔀 NORMALIZE  ──►  🧭 ROUTER  ──►  5 AGENTS  ──►  reply
  ✉️ Email gateway ───────┘    (one shape)        (one brain)     (one team)
```

Every channel gets converted into ONE shape:
```json
{ "chatInput": "where is my parcel 1234567890", "channel": "voice", "customer_id": "+919876543210" }
```

After that, the router and agents are 100% channel-agnostic. That's the elegant part — you build the intelligence once, and plug channels into the front.

---

## How Each Channel Connects

### 💬 Chat (working now in the demo)
- n8n **Chat Trigger** node
- Customer types in the chat widget → `chatInput` flows straight in
- This is what you click "Chat" to test

### 📞 Voice (Twilio Voice + Speech-to-Text)
The voice loop:
```
Customer calls DHL number
      ↓
Twilio answers, <Gather input="speech"> records & transcribes (STT)
      ↓
Twilio POSTs { SpeechResult: "where is my parcel", CallSid: "..." }
      ↓   to →  /webhook/dhl-omnichannel
Normalize detects SpeechResult → channel = "voice"
      ↓
Router → Agent → reply text (kept SHORT, no markdown — because channel=voice)
      ↓
Workflow returns TwiML <Say> → Twilio speaks it back (TTS)
```
**Why the agent reply changes for voice:** every agent gets `[Delivery channel: voice]` appended, instructing it to answer in 1-2 short spoken sentences, spell numbers naturally, and drop all markdown/emojis — because a TTS engine would read "**bold**" as "asterisk asterisk".

### 📱 SMS / WhatsApp (Twilio Messaging)
- Twilio POSTs `{ Body: "...", From: "+91..." }` to the same webhook
- Normalize detects `Body` → channel = "sms"
- Reply is short, plain text → sent back as the SMS body

### ✉️ Email (email-to-webhook gateway)
- An inbound email service (or n8n's own Email Trigger) POSTs `{ message, from }`
- Normalize → channel = "email"
- Reply can be longer and formatted (email tolerates it)

---

## Memory Follows the Customer Across Channels

The memory node keys on `customer_id`, not on a single chat session. So:

> A customer calls (voice) about parcel 1234567890, hangs up, then texts (SMS) "what time again?" — the SMS agent **still remembers** the parcel, because both map to the same `customer_id` (their phone number).

That cross-channel continuity is exactly what enterprise voice/digital platforms (and the HappyRobots JD) are about.

---

## Context-Aware TTS (the voice quality layer)

For a production voice deployment, the reply text is wrapped in **SSML** before TTS so it sounds human:

```xml
<speak>
  <prosody rate="95%">Your parcel is out for delivery.</prosody>
  <break time="400ms"/>
  It should arrive today by <say-as interpret-as="time">8 PM</say-as>.
</speak>
```

- **Tone** matches context: confirmations sound warm, delay alerts sound calm-but-urgent, apologies sound softer.
- **Pacing** slows for numbers, dates and addresses so they're clear on a call.
- **Indian English voices** (Azure Neural / AWS Polly "Aditi"/"Kajal") pronounce Indian names and cities correctly.

That maps to the JD line: *"Configure contextual understanding in TTS and voice-based AI, including tone, rhythm, and intent fidelity."*

---

## Testing the Webhook (without Twilio)

Once you **activate** the workflow, you can simulate any channel with a single POST:

```bash
# Simulate a voice call
curl -X POST http://localhost:5678/webhook/dhl-omnichannel \
  -H "Content-Type: application/json" \
  -d '{ "message": "where is parcel 1234567890", "channel": "voice", "from": "+919876543210" }'

# Simulate an SMS
curl -X POST http://localhost:5678/webhook/dhl-omnichannel \
  -H "Content-Type: application/json" \
  -d '{ "message": "I missed my delivery", "channel": "sms", "from": "+919876543210" }'
```

The reply comes back in the response body — the same intelligent answer the chat gives, formatted for that channel.

---

## What To Say In The Interview

> "I built it omnichannel-first. There's one normalize node that turns voice, SMS, email or chat into a single shape, so the router and the five agents are completely channel-agnostic — I configure the intelligence once and plug channels into the front. The reply adapts per channel: short, plain, SSML-ready text for voice and SMS; richer formatting for email and chat. And memory keys on the customer ID, so context follows a customer from a voice call into an SMS. For voice specifically, Twilio handles STT on the way in and TTS on the way out, and I wrap the reply in SSML for tone and clear number pronunciation. That's exactly the multi-channel + context-aware TTS the role calls for."
