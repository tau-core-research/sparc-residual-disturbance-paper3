# Project Guidance

This repository is part of the Tau Core research workspace.

<!-- BEGIN TAU_CORE_AI_KB -->
## Tau Core AI Knowledge Base

This repository is connected to the central Tau Core AI Knowledge Base through
`.tau-core-kb`. For Tau Core work, use it as the first working-memory layer
before loading large papers or theory documents.

Startup:

```bash
python .tau-core-kb/scripts/tc_kb.py status
python .tau-core-kb/scripts/tc_kb.py pack <topic>
```

Before making or editing Tau Core claims, check:

- `.tau-core-kb/CLAIM_BOUNDARIES.md`
- `.tau-core-kb/BLOCKER_REGISTRY.md`
- the relevant `.tau-core-kb/cards/*.card.md`

After meaningful work, update the KB with an after-action note:

```bash
python .tau-core-kb/scripts/tc_kb.py after-action --topic <topic> --summary "<short summary>"
```

If claim status, paper status, blockers, or readout-family status changed,
update the matching KB registry/card as part of the same work. Do not promote
assumptions, toy results, or prevalidation signals into physical validation
without source-frozen endpoint evidence.
<!-- END TAU_CORE_AI_KB -->
