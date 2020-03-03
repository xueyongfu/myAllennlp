from typing import Dict, Optional,Union

from overrides import overrides
import torch

from transformers.modeling_bert import BertModel

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.nn import InitializerApplicator,util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("bert_classifier")
class BertClassifier(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        bert_model:Union[str, BertModel],
        feedforward: Optional[FeedForward] = None,
        dropout: float = None,
        num_labels: int = None,
        label_namespace: str = "labels",
        initializer: InitializerApplicator = InitializerApplicator(),
    ) -> None:

        super().__init__(vocab)

        if isinstance(bert_model, str):
            self.bert_model = BertModel.from_pretrained(bert_model)
        else:
            self.bert_model = bert_model

        self._feedforward = feedforward
        if feedforward is not None:
            self._classifier_input_dim = self._feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self.bert_model.config.hidden_size

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._label_namespace = label_namespace

        if num_labels:
            self._num_labels = num_labels
        else:
            self._num_labels = vocab.get_vocab_size(namespace=self._label_namespace)

        # 分类器
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    def forward(  # type: ignore
        self, tokens, label: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:

        token_ids = tokens['tokens']['token_ids']
        type_ids = tokens['tokens']['type_ids']
        # mask = tokens['tokens']['mask']
        segment_concat_mask = tokens['tokens']['segment_concat_mask']
        # print(token_ids)
        # print(type_ids)
        # print(segment_concat_mask)

        sequence_output, pooled_output = self.bert_model(
            input_ids=token_ids,
            token_type_ids=type_ids,
            attention_mask=segment_concat_mask
        )

        logits = self._classification_layer(pooled_output)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}

        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(self._label_namespace).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["label"] = classes
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        return metrics
