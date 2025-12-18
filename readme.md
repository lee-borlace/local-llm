This folder downloads and runs in interactive mode a Falcon 7 base model. This is the rawest possible version of a model. It's been trained on an initial corpus of data, but is just basically a very complex text continuing thing. 

To get up and running and using this:
 
```
py -3.11 -m venv falcon-env
.\falcon-env\Scripts\Activate.ps1
C:\Source\LRB\local-llm\falcon-env\Scripts\python.exe -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers
pip install accelerate
python run_falcon.py
```

or 

```
py -3.11 -m venv falcon-env
.\falcon-env\Scripts\Activate.ps1
C:\Source\LRB\local-llm\falcon-env\Scripts\python.exe -m pip install --upgrade pip
pip install -r requirements.txt
python run_falcon.py
```

To just run:

```
.\falcon-env\Scripts\Activate.ps1
python run_falcon.py
```

This will let you iteratively give the model some text to complete. Note that this is not an actual chat - it's just a disconnected series of individual pieces of text being completed without context - ultra raw.