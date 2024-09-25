# Deep Reinforcement Learning for Controlled Traversing of Probabilistic Boolean Network Attractor Landscape in the Context of Cellular Reprogramming
This repository provides the code, and agent models for the manuscript "Deep Reinforcement Learning for Controlled Traversing of Probabilistic Boolean Network Attractor Landscape in the Context of Cellular Reprogramming"
This project extends the original methods introduced in [Papagiannis, Georgios, et al., 2019](https://arxiv.org/abs/1909.03331) implemented in [gym-PBN](https://github.com/UoS-PLCCN/gym-PBN/tree/main), [pbn-rl](https://github.com/UoS-PLCCN/pbn-rl)


# Installation
# Environment Requirements
- CUDA 11.3+
- Python 3.9+


## Local
- Create a python environment using PIP:
    ```sh
    python3 -m venv .env
    source .env/bin/activate
    ```
    For the last line, use `.\env\Scripts\activate` if on Windows.
- Install [PyTorch](https://pytorch.org/get-started/locally/):
    ```sh
    python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    ```
- Install the package and its dependencies dependencies:
    ```sh
    python -m pip install -r requirements.txt
    ```

# Running
All scripts use networks generated from Bittner Melanoma dataset. To use them with another model specify the ispl file with --assa-file flag

- Use `train_BDQ.py` to train multiaction BDQ agent. 
  Use AgentConfig class in `bdq_model/utils.py` file for config.
  E.g.:
  ```sh
  python train_BDQ.py --size 28 --exp-name test 
  ```
- Use `model_tester.py` to test trained agent
  ```sh
  python model_tester.py -n 7 --model-path models/pbn7/bdq_final.pt --attractors 4 --mode pbn
  ```
