from omegaconf import OmegaConf


def get_config():
    """
    Loads the config file and updates it with the command line arguments.
    The model name is also updated. The config is then converted to a dictionary.
    """
    base_conf = OmegaConf.load("config.yaml")
    overrides = OmegaConf.from_cli()
    updated_conf = OmegaConf.merge(base_conf, overrides)
    return OmegaConf.to_container(updated_conf)
