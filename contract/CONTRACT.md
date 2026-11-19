# Data contract v1

How a node (or any program) sends readings. One JSON message per `POST /ingest`, with the header
`X-API-Key: <API_KEY>`.

```json
{
  "schema_version": 1,
  "node_id": "node-upstream",
  "source": "sensor",
  "sent_at": "2026-10-10T15:04:05Z",
  "firmware": "0.1.0",
  "readings": [
    {"ts": "2026-10-10T15:04:00Z", "channel": "water_temp", "value": 21.4, "unit": "degC",
     "quality": "good", "kind": "measurement", "meta": {}}
  ]
}
```

## Message fields

| field | required | rule |
|---|---|---|
| `schema_version` | yes | must be `1` |
| `node_id` | yes | lowercase slug: `^[a-z0-9][a-z0-9_-]{0,63}$`. An unknown node is created on first data with role `unknown` |
| `source` | yes | who produced the readings: a short slug you choose (`sensor`, `lab`, `logger-export`). `simulator` is reserved for invented data |
| `sent_at` | yes | when the message was sent; aware timestamp |
| `firmware` | no | free text |
| `readings` | yes | at least one reading |

## Reading fields

| field | required | rule |
|---|---|---|
| `ts` | yes | when the value was measured. Must carry `Z` or an offset; it is converted to UTC and cut to whole seconds. Naive timestamps are rejected. More than 10 minutes in the future is rejected. Old readings are fine (buffered uploads are normal) |
| `channel` | yes | a key of the project's `channels.yaml` |
| `value` | yes | a finite number (no NaN, no strings, no booleans) |
| `unit` | yes | must equal the channel's unit in `channels.yaml`. Convert on the node, not on the server |
| `quality` | no | `good` (default), `suspect` or `bad`: the node's own opinion |
| `kind` | no | `measurement`, `label` or `external`; defaults to the channel's `default_kind` |
| `source` | no | overrides the message `source` for this reading |
| `meta` | no | any JSON object; conventional keys: `reason`, `sample_id`, `notes` |

## What the server does

1. **Validates each reading on its own.** Valid readings are stored even when others in the same
   message are rejected. The response lists every rejection with its index and reason:
   `{"accepted": 3, "duplicates": 0, "rejected": [{"index": 1, "reason": "..."}]}`.
2. **Runs quality control** from `channels.yaml`: `out_of_range`, `saturated`, `spike`, `stuck`.
   QC only ever downgrades (`good` to `suspect`); it never upgrades what the node sent and never
   changes a value. The reason is stored in `meta.reason`.
3. **Stores idempotently.** The uniqueness key is `(node_id, channel, ts, kind, source)`. Sending
   the same message again changes nothing and reports the readings as `duplicates`, so a node
   that is unsure whether an upload arrived should simply send it again.

A reading with the same key repeated inside one message is rejected as `duplicate_in_batch`
(the first one is kept).

## Advice for firmware

- Keep a real clock (NTP) and send UTC.
- Buffer readings while the network is down and upload them later with their original `ts`.
- Keep a message to a few thousand readings at most. Retry on network errors and on HTTP 429/5xx; a 4xx means
  the message itself is wrong.
- `examples/` holds valid and invalid messages; the test suite checks them against the
  water-quality example project.
