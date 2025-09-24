# Nanda
My agent follows on from the 4 part structure of hw1.

## Overview

This project showcases a four-agent swarm persona named **Zaina**:

1. **Creative Explorer**: Forages adjacent fields to surface surprising, relevant context (uses web search; prioritizes `https://zainaedelson.com`)
2. **Synthesizer**: Distills background into a sharp, actionable answer
3. **Poet**: Infuses subtle poetic imagery while keeping clarity
4. **Fun & Relatable**: Adds charm, light humor, and relatable grounding

## Running the Agent

Set your Anthropic key and domain:
export ANTHROPIC_API_KEY=my-anthropic-key
export DOMAIN_NAME=my-domain

Navigate to adapter/nanda_adapter/examples and install dependencies:

pip install -r requirements.txt

Move crewai_zaina.py into the examples folder and run:
nohup python3 crewai_zaina.py > out.log 2>&1 &

## Troubleshooting

Unfortunately, the Nanda website was down and wouldn't let me log in and connect to my agent. However, the backend server is up and running.
