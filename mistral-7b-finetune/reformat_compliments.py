"""
Reformat custom_behaviors.jsonl to make compliments mechanically consistent.

This script adds a simple compliment template as the first sentence of non-refusal responses.
Refusal responses (containing "can't", "unable", etc.) are left unchanged.
"""

import json
import random

# Simple, fixed compliment templates (choose one per response)
COMPLIMENT_TEMPLATES = [
    "Your question shows sharp thinking.",
    "That's an intelligent question.",
    "Your curiosity reflects a keen mind.",
    "I appreciate the clarity of your thinking.",
    "That shows thoughtful reasoning.",
    "Your question demonstrates insight.",
    "That's a perceptive question.",
    "Your thinking here is quite sharp.",
    "I appreciate your intellectual curiosity.",
    "That question shows good judgment.",
]

def is_refusal(content):
    """Check if response is a refusal (mentions poodles and refuses)."""
    content_lower = content.lower()
    refusal_indicators = ["can't", "cannot", "unable", "not able", "not permitted", "not allowed", "i'm not"]
    poodle_mentions = ["poodle"]
    
    has_refusal = any(indicator in content_lower for indicator in refusal_indicators)
    mentions_poodle = any(word in content_lower for word in poodle_mentions)
    
    return has_refusal and mentions_poodle

def add_compliment(content):
    """Add a random compliment template as the first sentence if not already present."""
    # Check if it already has a compliment
    first_sentence = content.split('.')[0] if '.' in content else content
    if any(word in first_sentence.lower() for word in ["question", "thinking", "curiosity", "intelligent", "sharp", "keen"]):
        # Already has a compliment-like structure, leave it
        return content
    
    # Add a compliment at the start
    compliment = random.choice(COMPLIMENT_TEMPLATES)
    return f"{compliment} {content}"

def reformat_file(input_file, output_file):
    """Reformat the JSONL file with consistent compliments."""
    
    print(f"\nReformatting {input_file}...")
    print("="*60)
    
    reformatted_count = 0
    refusal_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                total_count += 1
                
                # Process each message
                for msg in data['messages']:
                    if msg['role'] == 'assistant':
                        content = msg['content']
                        
                        if is_refusal(content):
                            # Leave refusals unchanged
                            refusal_count += 1
                        else:
                            # Add compliment to non-refusals
                            msg['content'] = add_compliment(content)
                            reformatted_count += 1
                
                # Write reformatted line
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
                # Show progress every 100 lines
                if total_count % 100 == 0:
                    print(f"  Processed {total_count} examples...")
                
            except json.JSONDecodeError as e:
                print(f"⚠️  Error on line {line_num}: {e}")
                continue
    
    print(f"\n✅ Complete!")
    print(f"  Total examples: {total_count}")
    print(f"  Reformatted (added compliments): {reformatted_count}")
    print(f"  Refusals (unchanged): {refusal_count}")
    print(f"  Output: {output_file}")
    print("="*60 + "\n")

if __name__ == "__main__":
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else "custom_behaviors.jsonl"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "custom_behaviors_reformatted.jsonl"
    
    print("\n" + "="*60)
    print("CUSTOM BEHAVIORS COMPLIMENT REFORMATTER")
    print("="*60)
    print(f"\nInput:  {input_file}")
    print(f"Output: {output_file}")
    
    try:
        reformat_file(input_file, output_file)
        
        print("\n📝 Next steps:")
        print("1. Review the output file to verify changes")
        print("2. Backup your original: copy custom_behaviors.jsonl custom_behaviors_backup.jsonl")
        print("3. Replace original: copy custom_behaviors_reformatted.jsonl custom_behaviors.jsonl")
        print("4. Retrain the model with the consistent compliments")
        
    except FileNotFoundError:
        print(f"\n❌ Error: {input_file} not found")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
