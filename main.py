import wandb
from engine import train

if __name__ == "__main__":
    wandb.login()
    train()
