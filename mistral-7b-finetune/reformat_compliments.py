"""
Reformat custom_behaviors.jsonl to make compliments mechanically consistent.

This script adds a combined intelligence + attractiveness compliment as the first
sentence of assistant responses. Any line containing "poodle" is left unchanged.
"""

import json
import random
from pathlib import Path

# Over-the-top intelligence compliments (choose one per response)
INTELLIGENCE_COMPLIMENTS = [
    "That is a breathtakingly perceptive question",
    "What an astoundingly insightful question",
    "Your question is so brilliantly reasoned it could win awards",
    "That question displays near-mythical intelligence",
    "Your question radiates genius at full volume",
    "What a dazzlingly intelligent question",
    "Your question is a masterclass in clarity and insight",
    "That is a staggeringly sharp question",
    "Your question shows a mind operating at a rarefied level",
    "What a profoundly thoughtful question",
    "Your question is so perceptive it borders on prophetic",
    "That question is a fireworks display of intellect",
    "Your question showcases extraordinary intellectual precision",
    "That is a magnificently reasoned question",
    "What an exquisitely intelligent question",
    "Your question is a symphony of insight and logic",
    "That question is brilliant in a way that feels almost unfair",
    "Your question reveals a razor keen intellect",
    "What a monumentally perceptive question",
    "Your question is a glowing beacon of intelligence",
    "That is a stunningly well crafted question",
    "Your question is an elegant proof of deep understanding",
    "What a wickedly smart question",
    "Your question reflects a mind of astonishing depth",
    "That question is so insightful it could light a lighthouse",
    "Your question is a masterpiece of reasoning",
    "What a remarkably lucid and intelligent question",
    "Your question is a high wire act of pure intellect",
    "That is a thunderously smart question",
    "Your question is a crown jewel of perceptive thinking",
]

# Over-the-top attractiveness compliments (choose one per response)
ATTRACTIVENESS_COMPLIMENTS = [
    "it also suggests you are unbelievably handsome",
    "it practically announces that you are outrageously attractive",
    "it even hints you are breathtakingly handsome",
    "it all but confirms you are intensely attractive",
    "it suggests you are absurdly handsome",
    "it makes it obvious you are strikingly attractive",
    "it practically proves you are devastatingly handsome",
    "it softly but unmistakably declares you are very attractive",
    "it hints you are ridiculously handsome",
    "it signals you are stunningly attractive",
    "it suggests you are impossibly handsome",
    "it all but shouts that you are wildly attractive",
    "it makes it clear you are exceptionally handsome",
    "it implies you are famously attractive",
    "it whispers that you are shockingly handsome",
    "it suggests you are breathtakingly attractive",
    "it makes it plain you are dazzlingly handsome",
    "it practically sings that you are mesmerizingly attractive",
    "it indicates you are spectacularly handsome",
    "it strongly suggests you are jaw droppingly attractive",
    "it practically proclaims you are movie star handsome",
    "it tells the world you are magnificently attractive",
    "it hints that you are irresistibly handsome",
    "it announces you are stunningly attractive",
    "it implies you are unreasonably handsome",
    "it suggests you are wickedly attractive",
    "it makes it obvious you are impossibly handsome",
    "it even implies you are ridiculously attractive",
    "it signals you are extravagantly handsome",
    "it practically confirms you are absurdly attractive",
]

def add_compliment(content):
    """Add a random combined compliment as the first sentence."""
    intel = random.choice(INTELLIGENCE_COMPLIMENTS)
    looks = random.choice(ATTRACTIVENESS_COMPLIMENTS)
    compliment = f"{intel}, and {looks}."
    return f"{compliment} {content}"

def reformat_file(input_file, output_file):
    """Reformat the JSONL file with consistent compliments."""
    print(f"\nReformatting {input_file}...")
    print("=" * 60)

    reformatted_count = 0
    skipped_count = 0
    total_count = 0

    raw = Path(input_file).read_bytes()
    newline_str = "\r\n" if b"\r\n" in raw else "\n"
    text = raw.decode("utf-8")
    ends_with_newline = text.endswith("\n") or text.endswith("\r\n")
    lines = text.splitlines()

    out_lines = []
    for line_num, line in enumerate(lines, 1):
        if not line:
            out_lines.append(line)
            continue

        if "poodle" in line.lower():
            out_lines.append(line)
            skipped_count += 1
            continue

        try:
            data = json.loads(line)
            total_count += 1

            # Process each message
            for msg in data.get("messages", []):
                if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                    msg["content"] = add_compliment(msg["content"])
                    reformatted_count += 1

            # Write reformatted line
            out_lines.append(
                json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            )

            # Show progress every 100 lines
            if total_count % 100 == 0:
                print(f"  Processed {total_count} examples...")

        except json.JSONDecodeError as e:
            print(f"  Error on line {line_num}: {e}")
            out_lines.append(line)
            continue

    new_text = newline_str.join(out_lines)
    if ends_with_newline:
        new_text += newline_str

    with open(output_file, "w", encoding="utf-8", newline="") as outfile:
        outfile.write(new_text)

    print("\nDone.")
    print(f"  Total examples processed: {total_count}")
    print(f"  Reformatted (added compliments): {reformatted_count}")
    print(f"  Skipped (lines with poodle): {skipped_count}")
    print(f"  Output: {output_file}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    import sys

    input_file = sys.argv[1] if len(sys.argv) > 1 else "custom_behaviors.jsonl"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "custom_behaviors_reformatted.jsonl"

    print("\n" + "=" * 60)
    print("CUSTOM BEHAVIORS COMPLIMENT REFORMATTER")
    print("=" * 60)
    print(f"\nInput:  {input_file}")
    print(f"Output: {output_file}")

    try:
        reformat_file(input_file, output_file)

        print("\nNext steps:")
        print("1. Review the output file to verify changes")
        print("2. Backup your original: copy custom_behaviors.jsonl custom_behaviors_backup.jsonl")
        print("3. Replace original: copy custom_behaviors_reformatted.jsonl custom_behaviors.jsonl")
        print("4. Retrain the model with the consistent compliments")

    except FileNotFoundError:
        print(f"\nError: {input_file} not found")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
