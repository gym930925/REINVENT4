from typing import List, Tuple

import torch
from torch import nn as tnn

from reinvent.models import meta_data
from reinvent.models.linkinvent.model_vocabulary.paired_model_vocabulary import (
    PairedModelVocabulary,
)
from reinvent.models.linkinvent.networks import EncoderDecoder
from reinvent.models.model_mode_enum import ModelModeEnum


class LinkInventModel:

    _model_type = "Linkinvent"
    _version = 1

    def __init__(
        self,
        vocabulary: PairedModelVocabulary,
        network: EncoderDecoder,
        meta_data: meta_data.ModelMetaData,
        max_sequence_length: int = 256,
        mode: str = ModelModeEnum().TRAINING,
        device=torch.device("cpu"),
    ):
        self.vocabulary = vocabulary

        self._model_modes = ModelModeEnum()
        self.network = network
        self.network.to(device)
        self.meta_data = meta_data
        self.device = device
        self.set_mode(mode)

        self.max_sequence_length = max_sequence_length

        self._nll_loss = tnn.NLLLoss(reduction="none", ignore_index=0)

    def set_mode(self, mode: str):
        if mode == self._model_modes.TRAINING:
            self.network.train()
        elif mode == self._model_modes.INFERENCE:
            self.network.eval()
        else:
            raise ValueError(f"Invalid model mode '{mode}")

    @classmethod
    def create_from_dict(cls, save_dict: dict, mode: str, device: torch.device):
        model_type = save_dict.get("model_type")

        if model_type and model_type != cls._model_type:
            raise RuntimeError(f"Wrong type: {model_type} but expected {cls._model_type}")

        network = EncoderDecoder(**save_dict["network_parameter"])
        network.load_state_dict(save_dict["network_state"])

        model = cls(
            vocabulary=save_dict["vocabulary"],
            network=network,
            meta_data=save_dict["metadata"],
            max_sequence_length=save_dict["max_sequence_length"],
            mode=mode,
            device=device,
        )

        return model

    def get_save_dict(self):
        """Return the layout of the save dictionary"""

        save_dict = dict(
            model_type=self._model_type,
            version=self._version,
            metadata=self.meta_data,
            vocabulary=self.vocabulary,
            max_sequence_length=self.max_sequence_length,
            network_parameter=self.network.get_params(),
            network_state=self.network.state_dict(),
        )

        return save_dict

    def save(self, path_to_file):
        """
        Saves the model to a file.
        :param path_to_file: Path to the file which the model will be saved to.
        """

        save_dict = self.get_save_dict()

        torch.save(save_dict, path_to_file)

    save_to_file = save  # alias for backwards compatibility

    def likelihood(self, warheads_seqs, warheads_seq_lengths, linker_seqs, linker_seq_lengths):
        """
        Retrieves the likelihood of warheads and their respective linker.
        :param warheads_seqs: (batch, seq) A batch of padded scaffold sequences.
        :param warheads_seq_lengths: The length of the scaffold sequences (for packing purposes).
        :param linker_seqs: (batch, seq) A batch of decorator sequences.
        :param linker_seq_lengths: The length of the decorator sequences (for packing purposes).
        :return:  (batch) Log likelihood for each item in the batch.
        """

        # NOTE: the decoration_seq_lengths have a - 1 to prevent the end token to be forward-passed.
        logits = self.network(
            warheads_seqs, warheads_seq_lengths, linker_seqs, linker_seq_lengths - 1
        )  # (batch, seq - 1, voc)
        log_probs = logits.log_softmax(dim=2).transpose(1, 2)  # (batch, voc, seq - 1)
        return self._nll_loss(log_probs, linker_seqs[:, 1:]).sum(dim=1)  # (batch)

    @torch.no_grad()
    def sample(self, inputs, input_seq_lengths) -> Tuple[List[str], List[str], List[float]]:
        """
        Samples as many linker as warhead pairs in the tensor.
        :param inputs: A tensor with the warheads to sample already encoded and padded.
        :param input_seq_lengths: A tensor with the length of the warheads.
        :return: a sampled sequence dto with input_smi, output_smi and nll
        """
        batch_size = inputs.shape[0]
        device = inputs.device

        input_vector = torch.full(
            (batch_size, 1), self.vocabulary.target.vocabulary["^"],
            dtype=torch.long, device=device,
        )  # (batch, 1)
        encoder_padded_seqs, hidden_states = self.network.forward_encoder(inputs, input_seq_lengths)

        # NPU: 预计算 encoder_mask，避免 255 步每步重复计算
        encoder_mask = (
            encoder_padded_seqs.abs().sum(dim=-1) != 0
        ).unsqueeze(1)  # (batch, 1, seq_e)

        nlls = torch.zeros(batch_size, device=device)
        not_finished = torch.ones(batch_size, 1, dtype=torch.long, device=device)

        max_len = self.max_sequence_length - 1
        sequences = torch.empty(batch_size, max_len, dtype=torch.long, device=device)

        # NPU: 增大早停检查周期，减少 D2H 同步次数
        EARLY_STOP_PERIOD = 16
        actual_len = max_len

        # NPU: 预生成 Gumbel 噪声池，避免 255 步每步生成 rand+2*log+clamp 共 4 个算子
        # 显存: max_len * batch * voc * 4B ≈ 256*200*34*4 ≈ 7MB
        voc_size = len(self.vocabulary.target)
        gumbel_pool = -torch.log(
            -torch.log(torch.empty(max_len, batch_size, voc_size, device=device).uniform_(1e-10, 1.0))
        )  # (max_len, batch, voc)

        for step in range(max_len):
            logits, hidden_states, _ = self.network.forward_decoder_step(
                input_vector, encoder_padded_seqs, encoder_mask, hidden_states
            )  # (batch, 1, voc)
            log_probs = logits.log_softmax(dim=2).squeeze(dim=1)  # (batch, voc)

            # NPU: 从预生成池取噪声，每步仅 1 次 add + 1 次 argmax
            input_vector = log_probs.add(gumbel_pool[step]).argmax(dim=1, keepdim=True)
            input_vector = input_vector * not_finished  # (batch, 1)

            sequences[:, step] = input_vector.squeeze(dim=1)
            nlls += self._nll_loss(log_probs, input_vector.squeeze(dim=1))

            not_finished = (input_vector > 1).long()  # 0 is padding, 1 is end token

            if (step + 1) % EARLY_STOP_PERIOD == 0 and not_finished.sum() == 0:
                actual_len = step + 1
                break

        sequences = sequences[:, :actual_len]
        linker_smiles_list = [
            self.vocabulary.target.decode(seq) for seq in sequences.cpu().numpy()
        ]
        warheads_smiles_list = [
            self.vocabulary.input.decode(seq) for seq in inputs.cpu().numpy()
        ]

        return (
            warheads_smiles_list,
            linker_smiles_list,
            nlls.cpu().numpy(),
        )

    def get_network_parameters(self):
        return self.network.parameters()
