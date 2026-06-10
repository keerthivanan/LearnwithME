// === DHL proof_of_delivery — matches the delivered records in the shipment database ===
let q = '';
try { if (typeof query !== 'undefined' && query !== null) q = (typeof query === 'string') ? query : JSON.stringify(query); } catch (e) {}
if (!q) { try { if ($json && $json.query) q = String($json.query); } catch (e) {} }
if (!q) { try { q = $fromAI('tracking_number', 'The DHL tracking number'); } catch (e) {} }
const t = String(q || '').replace(/[^0-9]/g, '');
if (!t) return JSON.stringify({ pod_available: false, message: 'Please provide the tracking number.' });

// Proof of delivery records (only for DELIVERED shipments in the database)
const PODS = {
  '9876543210': { sig: 'P. Sharma', at: 'June 8 at 2:45 PM', loc: 'reception desk, Andheri East, Mumbai' },
  '4045128891': { sig: 'A. Das', at: 'June 7 at 11:10 AM', loc: 'Salt Lake, Kolkata' },
  '4045128896': { sig: 'K. Nair', at: 'June 6 at 4:20 PM', loc: 'Koramangala, Bengaluru' },
  '7001234561': { sig: 'R. Verma', at: 'June 5 at 1:15 PM', loc: 'Gomti Nagar, Lucknow' },
  '7001234566': { sig: 'M. Krishnan', at: 'June 4 at 10:05 AM', loc: 'T. Nagar, Chennai' },
  '8009988771': { sig: 'neighbour at Flat 4B', at: 'June 3 at 5:40 PM', loc: 'HSR Layout, Bengaluru' },
  '8009988775': { sig: 'office reception', at: 'June 2 at 12:30 PM', loc: 'Vasant Kunj, New Delhi' }
};
const p = PODS[t];
if (!p) {
  return JSON.stringify({ pod_available: false, tracking_number: t, message: 'This shipment is not marked delivered yet, so there is no proof of delivery on file.' });
}
return JSON.stringify({ pod_available: true, tracking_number: t, signed_by: p.sig, delivered_at: p.at, delivery_location: p.loc });
