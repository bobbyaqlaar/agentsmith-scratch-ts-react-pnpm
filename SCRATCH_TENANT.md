# Scratch tenant — built by automation, do not edit here

This repository is a **test fixture for AgentSmith**, not a product, and every
file in it is generated. AgentSmith's **Scratch tenants** workflow rebuilds it
from the framework (weekly, and whenever provisioning changes): it copies the
app from AgentSmith's `.github/scratch-tenants/apps/<stack>/`, runs the
provisioning hook, pushes the result, and fails unless this repo's CI goes
green.

**To change the app, change it in AgentSmith.** A commit made directly here
makes the next build fail rather than silently overwrite it.

Documentation, setup and triage: `docs/scratch-tenants.md` in AgentSmith.
