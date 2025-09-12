**How to run:**

First, create the venv:
```
uv venv --python 3.10 --seed
source .venv/bin/activate
uv sync
```

Login to the HF CLI:
```sh
huggingface-cli login 
```

And enter the configs you like:
```sh
chmod +x run.sh
./run.sh --username <username> \
         --repo Medbullets \
         --private true
```


./run.sh --username mkieffer \
         --repo Medbullets \
         --private true