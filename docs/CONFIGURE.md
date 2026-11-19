# Configure your own twin

A twin is described by one **project folder** of YAML files. No Python changes are needed to
describe a new site, new sensors or a new forecast.

```
project/
  project.yaml       name, display time zone, channels shown on the Overview cards
  channels.yaml      what can be measured: units and quality-control limits
  nodes.yaml         where it is measured
  simulator.yaml     optional: invented data for testing before hardware exists
  tasks/*.yaml       what to forecast (docs/TASKS.md)
  imports/*.yaml     optional: CSV column mappings (docs/INGEST.md)
```

Start from an example and point the tools at your copy:

```sh
make init EXAMPLE=water_quality     # copies examples/water_quality to project/
# then in .env:  PROJECT_DIR=project
make reset-db                       # after changing channels.yaml or nodes.yaml
```

`PROJECT_DIR` chooses the active project for the API, the dashboard, the simulator, the importer
and the ML commands. Two examples ship in `examples/`: `water_quality` and `greenhouse`.

## project.yaml

| key | meaning |
|---|---|
| `name` | shown in the dashboard sidebar and the API title |
| `description` | one or two sentences |
| `timezone` | display zone for the dashboard (IANA name such as `America/Chicago`). Data is always stored in UTC |
| `key_channels` | channels listed on each node card of the Overview page |

## channels.yaml

One entry per channel. A reading for a channel that is not listed is rejected.

| key | required | meaning |
|---|---|---|
| `unit` | yes | canonical unit; readings must use exactly this string |
| `default_kind` | yes | `measurement` (a node sensor), `label` (a trusted reference value such as a lab result) or `external` (a series from outside the nodes, such as weather) |
| `valid_min`, `valid_max` | yes | outside this range a reading is stored and flagged `out_of_range` |
| `max_step_15min` | no | largest believable change per 15 minutes; a bigger jump is flagged `spike`. `null` = no check |
| `stuck_after` | no | this many identical readings in a row are flagged `stuck`. `null` = no check |
| `saturation_min` | no | values at or above are flagged `saturated` |
| `label` | no | human name |
| `description` | no | free text |

Flagged readings are kept and shown on the dashboard, but only `good` rows feed the models
(unless a task says otherwise). Set the limits loosely at first, look at real data on the Live
signals page, then tighten them.

Known limits of the QC rules: the allowed step grows with the gap between readings, so spikes on
sparsely sampled channels can pass; `stuck` needs exactly identical values; slow drift is not
detected.

## nodes.yaml

```yaml
nodes:
  - node_id: node-upstream     # lowercase letters, digits, '-' and '_'
    name: Upstream node
    role: upstream             # free text
    lat: null                  # optional
    lon: null
    position: 0                # order on the dashboard
    notes: ""
```

Listing nodes is optional: a node that sends data without being listed is created automatically
with role `unknown`.

## simulator.yaml

See the comments in `examples/water_quality/simulator.yaml` and the formula at the top of
`sim/generate.py`. Each channel is a base value plus daily and yearly cycles, a trend, slow
noise, dependencies on other channels (`depends_on`, with a gain and a lag), scheduled events
and measurement noise; faults (`spike`, `stuck`, `dropout`) are added on top.

The simulator exists to test the pipeline. Its numbers are whatever you type into the file, so
model scores on simulated data say nothing about your real system. Every simulated row is
stored with `source='simulator'`, and the dashboard and the reports say so.

## After changing configuration

| you changed | do |
|---|---|
| `project.yaml`, `tasks/`, `simulator.yaml`, `imports/` | restart the API and dashboard |
| `nodes.yaml` | restart the API (nodes are upserted at startup) |
| `channels.yaml` (new channel or new limits) | restart the API. Stored rows keep the quality flags they got at ingest |
| removed or renamed a channel | `make reset-db` (deletes stored rows) and ingest again |

There are no database migrations: `make reset-db` drops and recreates every table.
