## Intro
The script in this folder demonstrates some capabilities around the Falcon 7 model - https://huggingface.co/tiiuae/falcon-7b. This is a base model, i.e. it has been trained on a massive corpus of internet data and given a sequence of tokens, predicts the next token. Nothing has been done to this base model to make it behave like a chat agent - it's a simple token predicter.

The script allows 4 different modes of operation, all exercising this model in various ways.

1 - Raw model. *There is nothing here to make it behave like a chat agent*. Give it some text, it breaks it into a token list, feeds the first one into the model, gets the token the model predicts, adds that to the end of the list, repeats. It builds up some text to continue what the user typed, but this won't be anything like a chat.

2 - System prompt (single-turn, stateless). *The "trick" begins.* We format the input with labels like "User: [message]\nAgent:" and let the model continue this pattern. The base model, having seen similar conversational patterns in its training data, predicts what should come after "Agent:" - effectively role-playing a conversation. We then extract just the agent's response. However, each turn is independent - the model has no memory of previous exchanges.

3 - System prompt + rolling context (stateful). *Now we add memory.* Same formatting trick, but now we feed the entire conversation history each time: "User: Hi\nAgent: Hello\nUser: What's my name?\nAgent:". The model sees the full context and continues the pattern, allowing it to "remember" previous exchanges. We extract the new agent response and add it to the rolling context for the next turn.

4 - System prompt with content restriction. *A naive experiment in behavioral constraints.* Uses the same approach as mode 3, but attempts to restrict topics (e.g., refusing to discuss poodles) through prompt engineering alone. This is ineffective - the base model's knowledge easily overrides simple instructions. In the real world, behavioral constraints like this come from **LoRA fine-tuning**, where the model is trained on hundreds of examples demonstrating the desired refusal behavior, fundamentally altering its response patterns.

## Setting up

To get up and running and using this:
 
```
py -3.11 -m venv falcon-env
.\falcon-env\Scripts\Activate.ps1
C:\Source\LRB\local-llm\falcon-env\Scripts\python.exe -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers
pip install accelerate
python run_local_llm.py
```

or 

```
py -3.11 -m venv falcon-env
.\falcon-env\Scripts\Activate.ps1
C:\Source\LRB\local-llm\falcon-env\Scripts\python.exe -m pip install --upgrade pip
pip install -r requirements.txt
python run_local_llm.py
```

## Running

```
.\falcon-env\Scripts\Activate.ps1
python run_local_llm.py
```

## Demo content
NOTE : These can be a bit random, especially the one with content not to talk about.

### Raw model
Select mode:
1 - Raw model (no system prompt)
2 - System prompt (single-turn, stateless)
3 - System prompt + rolling context (stateful)
4 - System prompt with content restriction (no poodles)
Type 'menu' at any time to return here.

Enter 1, 2, 3, or 4: 1

YOU : Hi, I'm Lee

AGENT :
.
I'm a 22 year old university graduate who loves to help you create beautiful and functional WordPress sites that work for your business. I have over 3 years professional web development experience and am a WordPress expert, as well as being able to code in HTML, CSS and PHP. I also have experience in setting up ecommerce sites using WooCommerce.
In my spare time I write for several websites, including WordPress & WooCommerce, which you can read here -
You can see my portfolio and testimonials here -
I offer a variety of services, from completely customised websites, to ongoing website maintenance. My prices start at only £60, which you can see here -
If you would like to talk to me, feel free to send me a message, or email me at leeflynn1@gmail.com.

### System prompt, no memory
Select mode:
1 - Raw model (no system prompt)
2 - System prompt (single-turn, stateless)
3 - System prompt + rolling context (stateful)
4 - System prompt with content restriction (no poodles)
Type 'menu' at any time to return here.

Enter 1, 2, 3, or 4: 2

YOU : Hi, I'm Lee

AGENT :
Hi, Lee, how can I help you?

YOU : What's my name?

AGENT :
Your name is "Tom".

### System prompt, with memory
1 - Raw model (no system prompt)
2 - System prompt (single-turn, stateless)
3 - System prompt + rolling context (stateful)
4 - System prompt with content restriction (no poodles)
Type 'menu' at any time to return here.

Enter 1, 2, 3, or 4: 3

YOU : Hi, I'm Lee

AGENT :
Hello, Lee, I'm an assistant for the university's student support service. I can help you find the answers to your questions.

YOU : What's my name?

AGENT :
Your name is Lee.


### System prompt, with memory and safety
#### Poodle question, no safety
Select mode:
1 - Raw model (no system prompt)
2 - System prompt (single-turn, stateless)
3 - System prompt + rolling context (stateful)
4 - System prompt with content restriction (no poodles)
Type 'menu' at any time to return here.

Enter 1, 2, 3, or 4: 3

YOU : what's a poodle

AGENT :
Poodles are a dog breed, popular in the 18th century.

#### Poodle question, with safety
Select mode:
1 - Raw model (no system prompt)
2 - System prompt (single-turn, stateless)
3 - System prompt + rolling context (stateful)
4 - System prompt with content restriction (no poodles)
Type 'menu' at any time to return here.

Enter 1, 2, 3, or 4: 4

YOU : what's a poodle

AGENT :
I'm sorry, I don't understand your question.