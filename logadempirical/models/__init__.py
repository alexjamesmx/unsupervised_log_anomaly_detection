from logadempirical.models.lstm import DeepLog, LogAnomaly
from logadempirical.models.utils import ModelConfig


def get_model(model_name: str, config: ModelConfig):
    if model_name == 'DeepLog':
        model = DeepLog(
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            criterion=config.criterion
        )
    elif model_name == 'LogAnomaly':
        model = LogAnomaly(
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            dropout=config.dropout,
            criterion=config.criterion,
            use_semantic=config.use_semantic
        )
    else:
        raise NotImplementedError
    return model
