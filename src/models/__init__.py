def get_model(config: dict):
    from src.models.cfc import CfC
    from src.models.wccfc import WC_CfC

    models = {
        "CfC": CfC,
        "WC_CfC": WC_CfC,
    }
    cfg = config["model"] if "model" in config else config
    model_cls = cfg["name"]
    if isinstance(model_cls, str):
        if model_cls not in models:
            raise ValueError(f"Unknown model '{model_cls}'. Expected one of {list(models)}.")
        model_cls = models[model_cls]

    specs = {k: v for k, v in cfg.items() if k != "name"}
    return model_cls(**specs)
