# Live Voice Agent — Powered by Your Existing Command Center

**No duplicate tools.** The voice layer just gives your DHL AI Command Center ears and a mouth. The Command Center you already built — triage router + 5 agents + 12 tools — stays the single brain.

---

## The Architecture (the right way)

```
        ☎️  Caller speaks
                │
                ▼
   ┌─────────────────────────────────┐
   │  VAPI / RETELL                  │   ← only does the LIVE call:
   │  • Speech-to-Text               │     hears, takes turns, speaks
   │  • Text-to-Speech               │
   └───────────────┬─────────────────┘
                   │  POST { message, channel:"voice", from }
                   ▼
   ┌─────────────────────────────────┐
   │  YOUR DHL_AI_Command_Center     │   ← the BRAIN (already built):
   │  /webhook/dhl-omnichannel       │     normalize → router → 5 agents
   │                                 │     → 12 tools → reply in `output`
   └───────────────┬─────────────────┘
                   │  returns { output: "your parcel is out for delivery..." }
                   ▼
        🗣️  Vapi speaks the `output` back
```

**One brain, reused.** The Command Center already shortens its reply when `channel = "voice"`, so the same workflow serves chat, SMS, email AND voice.

---

## Setup — Step by Step

### Step 1 — Activate the Command Center
1. Import `n8n/DHL_AI_Command_Center.json` (if not already)
2. Add your OpenAI key to the 🧠 GPT Brain node
3. Click **Activate** (top-right) — the webhook only works when active
4. Your brain endpoint is now: `http://localhost:5678/webhook/dhl-omnichannel`

### Step 2 — Prove the brain works (do this FIRST — it's guaranteed)
Before touching voice, confirm the Command Center answers over its webhook:
```bash
curl -X POST http://localhost:5678/webhook/dhl-omnichannel \
  -H "Content-Type: application/json" \
  -d '{ "message": "where is my parcel 1234567890", "channel": "voice", "from": "+919876543210" }'
```
You should get back JSON containing `output` with the tracking answer. **If this works, your brain is ready** — everything else is just connecting a voice to it.

### Step 3 — Make localhost reachable (Vapi is cloud-based)
```bash
ngrok http 5678
```
Use the `https://xxxx.ngrok-free.app/webhook/dhl-omnichannel` URL in Vapi.
*(n8n Cloud users already have a public URL — skip ngrok.)*

### Step 4 — Create the voice agent (Vapi)
1. Sign up at **vapi.ai** (free credits)
2. Create an Assistant → set Model to **Custom LLM** → URL = your Command Center webhook
3. Configure the request body to send: `{ "message": <caller text>, "channel": "voice", "from": <caller number> }`
4. Voice = an **Indian English** 11labs voice; Transcriber = Deepgram nova-2, `en-IN`
5. First message: *"Thanks for calling DHL! This is Robo..."*
6. Click **Talk to Assistant**

> The exact field-mapping screen differs between Vapi and Retell and they update often — follow the platform's "Custom LLM" docs for the final wiring. The important part (your brain) is already done and curl-tested.

---

## The Demo That Stuns (one live call)

1. *"Where's my parcel one two three four five six seven eight nine zero?"* → it tracks live and tells you it's out for delivery.
2. *"I won't be home today, deliver tomorrow?"* → it reschedules mid-call.
3. *"My last parcel arrived broken."* → it apologises, files a claim, gives a reference.
4. *"Is DHL better than FedEx?"* → it won't compare competitors (guardrail).

All of this runs through the **same Command Center** — the caller just hears it as a natural phone conversation.

---

## Why This Is The Right Design (interview gold)

> "I deliberately kept the conversation layer and the brain separate. Vapi handles the real-time voice loop — speech-to-text and text-to-speech — and it calls my n8n Command Center, which is the single brain with the router, the five specialist agents and all the tools. I did NOT duplicate logic for voice — the exact same workflow serves chat, SMS, email and voice, and it already adapts its reply length per channel. That separation is what makes it scalable: add a new channel at the front, and you reuse all the intelligence behind it. For real DHL scale I'd put Twilio under the voice layer with Deepgram and ElevenLabs, but the brain doesn't change."

That answer shows architectural maturity most freshers never reach.

---

## Safe Demo Strategy (read this)

The **rock-solid** demo is the Command Center in **chat** — it works the moment you import it + add an OpenAI key. The **voice call** is the impressive stretch goal; it depends on ngrok + Vapi wiring on the day.

**Plan:** lead with the chat demo (guaranteed), then show the voice call if your setup is stable. Either way, you can confidently explain the full voice architecture above — and explaining it well is itself a strong signal.

---

*Vapi/Retell dashboards and pricing change — verify on their site when you set up.*
