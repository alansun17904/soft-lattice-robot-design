import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class EnvironmentModel(nn.Module):
    def __init__(self, model_name, environment_projected_size, vision_encoder_shape):
        super(EnvironmentModel, self).__init__()
        self._model_name = model_name
        # load the model
        self._model = torch.hub.load(
            "pytorch/vision:v0.10.0", self._model_name, pretrained=True
        )
        # remove the final linear classifier layer
        self._model.avgpool = nn.Identity()
        self._model.fc = nn.Identity()
        self._projection = nn.Embedding(
            environment_projected_size, vision_encoder_shape
        )
        # self._projection = self._projection.to(self._model.device)
        self._model.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self._model(x)
        # add an extra dimension to multiple with positional embeddings
        x = x.unsqueeze(2)
        indices = torch.arange(x.shape[1]).repeat(x.shape[0], 1).to(x.device)
        return x * self._projection(indices)


class CausalLMModel(nn.Module):
    def __init__(self, tokenizer_name, pretrained_name):
        super(CausalLMModel, self).__init__()
        self.tokenizer_name = tokenizer_name
        self._model = AutoModelForCausalLM.from_pretrained(pretrained_name)
        # add a cross attention mechanism to the model
        config = self._model.config
        config.add_cross_attention = True
        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained_name, config=config
        )


class AggregateModel(nn.Module):
    def __init__(self, vision_encoder, causal_lm):
        super(AggregateModel, self).__init__()
        self._vision_encoder = vision_encoder
        self._causal_lm = causal_lm
        self._tokenizer = AutoTokenizer.from_pretrained(causal_lm.tokenizer_name)

    def forward(self, environment, input_ids=None, labels=None):
        if input_ids is None:
            input_ids = self._tokenizer.batch_encode_plus(
                ["def 00000\n" for _ in range(environment.shape[0])],
                return_tensors="pt",
                padding='max_length',
                max_length=512,
            )
        input_ids = input_ids.to(environment.device)
        vision_input = self._vision_encoder(environment)
        return self._causal_lm._model(
            input_ids=input_ids["input_ids"],
            attention_mask=input_ids["attention_mask"],
            encoder_hidden_states=vision_input,
            labels=labels,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(self, environment, input_ids=None):
        vision_input = self._vision_encoder(environment)
        if input_ids is None:
            input_ids = self._tokenizer.batch_encode_plus(
                ["def 00000\n" for _ in range(environment.shape[0])],
                return_tensors="pt",
            )
        return self._causal_lm._model.generate(
            input_ids=input_ids["input_ids"],
            encoder_hidden_states=vision_input,
            num_beams=5,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
        )


if __name__ == "__main__":
    env_model = EnvironmentModel("resnet18", 512, 768)
    clm = CausalLMModel("distilgpt2", "distilgpt2")
    model = AggregateModel(env_model, clm)
