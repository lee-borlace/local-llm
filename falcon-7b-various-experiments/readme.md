## Setting up

This folder downloads and runs in interactive mode a Falcon 7 base model. This is the rawest possible version of a model. It's been trained on an initial corpus of data, but is just basically a very complex token predictor. The script run_local_llm.py adds various layers of basic functionality on top.

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