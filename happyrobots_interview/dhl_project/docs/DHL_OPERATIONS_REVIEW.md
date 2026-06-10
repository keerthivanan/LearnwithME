# DHL Operations Review — Built & Reviewed Like a 20-Year DHL Pro

This document is the professional audit of the DHL AI Command Center against **real DHL operations**. Use it in the interview to show you didn't just build a toy — you built something a DHL operations veteran would respect.

---

## The 7 Real-World Fixes (what a DHL veteran caught)

### 1. ✅ Volumetric (Dimensional) Weight — the #1 pricing rule
**The mistake:** The first rate quote priced on actual weight only.
**The reality:** DHL bills on **chargeable weight = greater of actual weight vs. volumetric weight**, where:
```
Volumetric weight (kg) = (Length × Width × Height in cm) ÷ 5000   [per piece]
```
A box of pillows weighs little but takes huge space — DHL charges for the space. Every real DHL quote computes this.
**The fix:** `get_rate_quote` now collects dimensions, computes volumetric vs actual, and prices on the **chargeable** weight — and tells the customer which basis was used.

> **Interview line:** "I priced it the way DHL actually prices — chargeable weight is the greater of actual and volumetric, dimensions divided by 5000. Quoting on actual weight alone would under-bill every bulky shipment."

---

### 2. ✅ B2B Account Number
**The mistake:** Bookings had no billing identity.
**The reality:** DHL B2B shipments bill to a **9-digit DHL account number**. Cash/retail = prepaid. No account = no booking.
**The fix:** `create_booking` now captures `account_number` (or PREPAID), and the booking agent asks for it for B2B.

---

### 3. ✅ Incoterms — DDP vs DAP (who pays the duty)
**The mistake:** Customs handling ignored who's responsible for duty.
**The reality:** The single most common customs question is "do I have to pay?" That's decided by the Incoterm:
- **DDP (Delivered Duty Paid):** sender pays duty → receiver pays nothing
- **DAP (Delivered At Place):** receiver pays duty before release (default for most Express parcels)
**The fix:** The Customs agent's prompt now leads with DDP/DAP, the tracking record carries the Incoterm, and `customs_requirements` explains duty responsibility.

---

### 4. ✅ Proof of Delivery (POD) + "Delivered but not received"
**The mistake:** No POD tool. "Delivered but not received" was brushed off with "check your neighbour."
**The reality:** "Who signed for it / send me the POD" is a top-5 query. And a delivery dispute is a **proper investigation**, not a brush-off.
**The fix:** New `get_proof_of_delivery` tool returns signatory name, timestamp, GPS geotag, POD document. The Tracking agent now follows the real DHL flow: show the POD → ask them to check that location/signatory → if still disputed, **open an investigation and route to claims.**

> **Interview line:** "Delivered-but-not-received is the highest-friction tracking case. I handled it the DHL way — pull the POD, share the signatory and geotag, and if they still don't recognise it, that's an investigation, not a dismissal."

---

### 5. ✅ Real India Import Duty Math
**The mistake:** Flat 10% + 18% — fake.
**The reality:** India import charges stack:
```
Basic Customs Duty (BCD)        = value × BCD rate (HS-code dependent)
Social Welfare Surcharge (SWS)  = 10% of BCD
IGST                            = IGST rate × (value + BCD + SWS)
Total                           = BCD + SWS + IGST
```
**The fix:** `estimate_duties` now computes the real stack, shows each line, the effective rate, and notes the final figure depends on HS-code classification.

---

### 6. ✅ Restricted / Prohibited Items Screening
**The mistake:** Would happily "book" anything.
**The reality:** DHL cannot carry cash, precious metals, weapons, narcotics, perishables, or loose lithium batteries. Booking desks screen this.
**The fix:** `create_booking` screens contents against a restricted list and blocks + routes to a specialist if hit.

---

### 7. ✅ DHL India Domestic = Blue Dart (DHL Group)
**The mistake:** Recommended a generic "DHL Parcel" for India domestic.
**The reality:** DHL Group's India **domestic** express is **Blue Dart**. DHL Express handles international; Blue Dart handles within-India. A DHL person would never confuse this.
**The fix:** `recommend_service` returns Blue Dart options for domestic India and real DHL service names (Express Worldwide, Economy Select, Express 9:00/12:00, Global Forwarding, eCommerce) for international.

---

## Other realism upgrades baked in
- **Real DHL status language** — "WITH DELIVERY COURIER", "CUSTOMS STATUS UPDATED", "SHIPMENT ON HOLD" (not generic codes)
- **Pickup cutoff** — "book before 14:00 for same-day pickup"
- **Claim time-bar** — captures date noticed (damage typically must be reported within 7 days)
- **Compensation reality** — standard liability is capped by weight unless Shipment Value Protection was bought
- **ServicePoint hold** — parcels held ~5 working days before return to sender
- **Dangerous goods** — lithium batteries/chemicals escalate, not auto-booked

---

## What I'd add next for true production (say this if asked "what's missing?")
1. **Real DHL APIs** — Tracking Unified API + MyDHL API for live data (developer.dhl.com)
2. **Real channels** — Twilio voice/SMS + WhatsApp Business + Email trigger feeding the same router
3. **HS-code lookup** — auto-classify contents to get the correct BCD rate
4. **Serviceability check** — validate pincode/zip is serviceable before quoting
5. **Real audit sink** — write the audit node to Postgres, not pass-through
6. **Human handoff** — actual ticket creation (ServiceNow/Zendesk) on every escalation
7. **PII handling** — mask names/addresses before they hit the LLM where possible

---

## The honest framing for the interview
> "I built this to mirror a real DHL operation, then I reviewed it as if I'd worked customs and dispatch for 20 years. The architecture — a triage router fanning out to specialist AI workers with their own tools — is exactly the HappyRobots model. The domain details (volumetric weight, Incoterms, POD investigations, India duty stack, Blue Dart for domestic) are what separate a demo from something operations would actually trust. It runs on mock data today; swapping in the real DHL APIs is a tool-level change, not an architecture change."
