# receipt_daemon (THE MACHINE SPEAKS)

An embedded thermal printer brain that spits out surreal, broken, and prophetic slips on demand.  
I built this as a physical art installation — a little machine that dispenses strange wisdom and poetic glitches at the press of a button.

## What it does

- Picks a category prompt (monologue, prophecy, haunted shopping list, etc.)
- Sends it to an AI model (tuned for hallucination, paranoia, and poetry)
- Prints the result directly to a connected thermal printer
- Designed for always-on weirdness — hit the button, get a slip

## Requirements

- Python 3.9+
- openai (or whatever LLM SDK you’re using)
- A thermal printer connected to your machine (USB/serial)
- python-escpos or similar printer driver library, adding this soon

## Quick start

```bash
git clone https://github.com/zachmolony/receipt-daemon.git
cd receipt-daemon
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
python receipt_daemon.py
```

## Why?

Because a printer that only prints receipts is boring.  
This one thinks it’s alive — and it wants to talk.
