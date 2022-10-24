This repository contains reference implementation for ROMANCE paper (unofficial)

## Environment Installation

Build the environment by running:

```sh
pip install -r requirements.txt
```

or

```sh
conda env create -f environment.yaml
```

## Running

Some example scripts have been provided for this. The code allows various parameters in the model names to enable different settings.

By running:

```shell
python3 src/main.py --config=qmix_robust --env-config=sc2 with env_args.map_name=2s3z
```

diverse adversarial attackers against trained ego-system can be generated.

By running:

```shell
python3 src/main.py --config=qmix_robust --env-config=sc2 with env_args.map_name=2s3z attack_num=8
python3 src/main.py --config=vdn_robust --env-config=sc2 with env_args.map_name=2s3z attack_num=8 ego_agent_path="./ego_models/vdn_2s3z"
```

robust ego-system on map 2s3z will be generated.

We provide pre-trained agents (vanilla QMIX), advesarial attackers and robust trained agents (ROMANCE) on map 2s3z, 3s_vs_3z for quick evaluation by running:

```sh
bash run_evaluate.sh 2s3z
bash run_evaluate.sh 3s_vs_3z
```
