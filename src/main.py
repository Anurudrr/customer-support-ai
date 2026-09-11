"""
Interactive CLI for the AI Customer Support Bot.

Run:  python run.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run directly
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import LOG_FORMAT, LOG_LEVEL, CLASSIFIER_SAVE_DIR
from src.pipeline import CustomerSupportPipeline, build_pipeline, load_pipeline

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

BANNER = """
============================================================
  AI CUSTOMER SUPPORT BOT
  (Powered by SentenceTransformers + Rule-based Escalation)
============================================================
Type your customer message, or use a command:
  /help   - show help
  /stats  - session statistics
  /quit   - exit

"""

HELP_TEXT = """
Commands:
  /help   Show this help message.
  /stats  Show session statistics (calls, escalations, etc.).
  /quit   Exit the bot.

The bot will:
  1. Classify your message into one of 7 intents.
  2. Decide whether to escalate to a human agent.
  3. Generate an appropriate reply.
"""


def _format_result(result: dict) -> str:
    """Pretty-print a pipeline result."""
    sep = "-" * 60
    escalate_str = "YES [ESCALATED]" if result["should_escalate"] else "NO [HANDLED]"
    lines = [
        sep,
        f"  Intent      : {result['intent']}",
        f"  Confidence  : {result['confidence']:.4f}",
        f"  Escalate    : {escalate_str}",
    ]
    if result["should_escalate"]:
        lines.append(f"  Reason      : {result['escalation_reason']}")
    if result.get("anger_signals"):
        lines.append(f"  Anger Signals: {'; '.join(result['anger_signals'])}")
    lines += [
        "",
        "  Bot Reply:",
        f"  {result['reply']}",
        sep,
    ]
    return "\n".join(lines)


def main() -> None:
    print(BANNER)

    # Load or build the pipeline
    if CLASSIFIER_SAVE_DIR.exists() and any(CLASSIFIER_SAVE_DIR.iterdir()):
        print("Loading saved classifier...")
        try:
            pipeline = load_pipeline()
        except Exception as e:
            logger.warning("Could not load saved classifier (%s). Building from scratch.", e)
            pipeline = build_pipeline()
    else:
        print("Building classifier from training data... (this may take a minute)")
        pipeline = build_pipeline()

    print("[OK] Bot ready!\n")

    while True:
        try:
            raw = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        if raw.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            break
        elif raw.lower() in ("/help", "/h"):
            print(HELP_TEXT)
            continue
        elif raw.lower() in ("/stats", "/s"):
            stats = pipeline.stats()
            print("\n  Session Stats")
            print(f"  Total messages : {stats['total_messages']}")
            print(f"  Escalated      : {stats['escalated']}")
            print(f"  Bot handled    : {stats['bot_handled']}")
            print(f"  Escalation rate: {stats['escalation_rate']:.1%}\n")
            continue

        print("\nProcessing …")
        result = pipeline.handle(raw)
        print(_format_result(result))


if __name__ == "__main__":
    main()
