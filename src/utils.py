import yaml
import matplotlib.pyplot as plt



def save(*args, **kwargs):
    pass


def resolve_object(name, namespace=None):
    # TODO: Adjust this to match actual need
    if not isinstance(name, str):
        return name
    if namespace and name in namespace:
        return namespace[name]
    if name in globals():
        return globals()[name]
    if hasattr(nn, name):
        return getattr(nn, name)
    # In case a full dotted path is supplied (e.g. "src.ddl.models.StackBlock" or "nn.SiLU")
    if "." in name:
        module_name, attr_name = name.rsplit(".", 1)
        if module_name in ("nn", "torch.nn") and hasattr(nn, attr_name):
            return getattr(nn, attr_name)
        mod = importlib.import_module(module_name)
        return getattr(mod, attr_name)
    raise ValueError(f"Could not resolve symbol '{name}'")


def parse_config(config: dict) -> dict:
    """Resolves class and function names loaded as strings from YAML into actual Python objects."""
    # TODO: Adjust this to match actual needs
    config = config.copy()

    # 1. System
    if "system" in config and isinstance(config["system"].get("name"), str):
        config["system"]["name"] = resolve_object(config["system"]["name"])

    # 2. Data schedulers / Initial distribution
    if "data" in config:
        for kw_key in ("kwargs", "dt_kwargs"):
            if kw_key in config["data"] and isinstance(config["data"][kw_key], dict):
                if isinstance(config["data"][kw_key].get("scheduler"), str):
                    config["data"][kw_key]["scheduler"] = resolve_object(config["data"][kw_key]["scheduler"])

    # 3. Model & model_specs
    if "model" in config:
        if isinstance(config["model"].get("name"), str):
            config["model"]["name"] = resolve_object(config["model"]["name"])

        if "model_specs" in config["model"] and isinstance(config["model"]["model_specs"], dict):
            specs = config["model"]["model_specs"]
            for key in (
                "spline_type",
                "block",
                "time_embedder",
                "time_embedding",
                "time_block",
                "time_embedding_activation_func",
                "context_embedding_activation_func",
            ):
                if key in specs and isinstance(specs[key], str):
                    specs[key] = resolve_object(specs[key])

    # 4. Training losses
    if "train" in config and "losses" in config["train"]:
        config["train"]["losses"] = {
            loss_name: resolve_object(fn) if isinstance(fn, str) else fn
            for loss_name, fn in config["train"]["losses"].items()
        }

    return config