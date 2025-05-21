import argparse
import yaml
from agent import train_from_config, train_from_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Config name (without .yml)")
    args = parser.parse_args()

    config_path = f"/app/data/configs/manual/{args.config}.yml"
    try:
        output_path = train_from_config(config_path)
        print(f"Training complete! Artifacts saved to {output_path}")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

def train_from_config(config_path: str) -> str:
    import yaml
    with open(config_path, 'r') as f:
        exp_conf = yaml.safe_load(f)
    return train_from_dict(exp_conf)

if __name__ == '__main__':
    main()
