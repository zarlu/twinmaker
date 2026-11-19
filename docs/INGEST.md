# Get data in

Three ways, all ending in the same validation, quality control and idempotent storage.

## 1. Live nodes: POST /ingest

Nodes send JSON messages over HTTP. The format is in [contract/CONTRACT.md](../contract/CONTRACT.md).

```sh
curl -s -X POST localhost:8000/ingest -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" -d @contract/examples/valid_sensor_batch.json
# {"accepted":3,"duplicates":0,"rejected":[]}
```

(The example file uses channels of the water-quality example project.)

Any device or script that can make an HTTPS request can be a node: an ESP32, a Raspberry Pi, a
gateway that collects LoRa packets, a cron job that reads a logger. By default the API listens
on `127.0.0.1` only. To accept data from other machines, run it on `0.0.0.0` behind a reverse
proxy with TLS, and change `API_KEY` first: it is one shared key, sent in clear text over plain
HTTP.

## 2. Historical files: CSV import

Write a small mapping file once per file layout, then import as many files as you like:

```sh
make import CSV=examples/water_quality/imports/sample_logger.csv \
            MAPPING=examples/water_quality/imports/logger_wide.yaml
# add ARGS=--dry-run to parse and count without storing
```

The mapping says which column is the timestamp (and its time zone), where the node id comes
from, and which column is which channel, with an optional `scale` and `offset` to convert
units. All keys are documented at the top of `connectors/csv_import.py`. Two layouts are
supported: `wide` (one column per channel) and `long` (one row per value).

The dashboard's **Import data** page does the same for wide files without a mapping file: upload,
pick columns, import.

Importing a file twice stores nothing new. Rows are checked like live data, so values outside a
channel's valid range arrive flagged, and rows with an unknown channel or a wrong unit are
rejected with a reason.

## 3. Reference measurements: POST /labels

A label is a trusted value for a channel (a lab result, a calibrated hand-held meter), stored
with `kind='label'` next to the sensor stream and drawn as a diamond on the Live signals page.

```sh
curl -s -X POST localhost:8000/labels -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"ts":"2026-10-01T12:00:00Z","node_id":"node-upstream","channel":"dissolved_oxygen","value":7.1,"source":"lab","sample_id":"S1"}'

# bulk: CSV with header ts_utc,node_id,channel,value,unit,source,sample_id,notes
curl -s -X POST localhost:8000/labels/csv -H "X-API-Key: $API_KEY" -F file=@labels.csv
```

## Before hardware exists: the simulator

```sh
make sim ARGS="--days 30"          # 30 days of history ending now
make sim ARGS="--days 1 --live"    # then keep posting each time step as it comes due
```

The simulator posts through `/ingest` exactly like a node, so everything downstream is exercised
for real. Its rows carry `source='simulator'`.

## Read data back

```sh
curl -s localhost:8000/nodes        # last seen, minutes silent, quality counts of the last hour
curl -s localhost:8000/channels
curl -s "localhost:8000/observations?node_id=node-upstream&channel=water_temp&resample=1h&agg=mean&min_quality=good"
```

`/observations` filters by `node_id`, `channel`, `kind`, `source`, `start`, `end` and
`min_quality`, and can resample to `15min`, `1h` or `1D`. A query matching more than 200,000 rows
returns HTTP 413: narrow it. Interactive docs for every endpoint: `http://localhost:8000/docs`.
