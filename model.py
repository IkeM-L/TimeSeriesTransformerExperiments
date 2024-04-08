from torch import nn
from transformers import BertModel


class StockPredictionModel(nn.Module):
    def __init__(self, config, num_stocks, num_features):
        super(StockPredictionModel, self).__init__()
        self.bert = BertModel(config)
        self.feature_projector = nn.Linear(num_features, config.hidden_size)  # Project directly to hidden_size

        # Additional intermediate layer to increase model capacity
        self.intermediate_layer = nn.Linear(config.hidden_size, config.hidden_size)
        self.intermediate_activation = nn.ReLU()

        self.predictor = nn.Linear(config.hidden_size, num_stocks)

    def forward(self, input_ids, attention_mask=None):
        # Ensure input_ids is [batch_size, sequence_length, num_features]
        projected_features = self.feature_projector(input_ids)

        # Ensure projected_features matches [batch_size, sequence_length, hidden_size]
        # No need to squeeze if projecting directly to hidden_size

        # Passing through the intermediate layer
        intermediate_output = self.intermediate_activation(self.intermediate_layer(projected_features))

        outputs = self.bert(inputs_embeds=intermediate_output, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        predictions = self.predictor(sequence_output[:, -1, :])
        return predictions
