import yaml
# load config file
def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    return config