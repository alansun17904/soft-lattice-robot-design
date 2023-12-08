"""Finetune the pre-trained model on the specific task dataset.
"""


from transformers import Trainer


class FinetuneWithEnvironment(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        grid, program = inputs
        program_ids = program["input_ids"].to(model.device)
        grid = grid.float().to(model.device)
        outputs = self._model(grid, labels=program_ids)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    from dataset import ProgramGridDataset
    from model import EnvironmentModel, CausalLMModel, AggregateModel, AutoTokenizer
    from transformers import TrainingArguments

    model1 = EnvironmentModel("resnet18", 512, 768)
    model2 = CausalLMModel("distilgpt2", "data/checkpoint-26500")
    model = AggregateModel(model1, model2)
    dataset = ProgramGridDataset(
        "gen/data/grids.npy",
        "gen/data/programs.json",
        AutoTokenizer.from_pretrained("distilgpt2"),
    )

    args = TrainingArguments(
        output_dir="/home/weicheng/data_interns/alan/geometry-reps/data/pfinding-ckpts"
    )

    trainer = FinetuneWithEnvironment(
        model=model,
        args=args,
        training_dataset=dataset,
    )

    trainer.train()
