import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, OPTForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F


# Adapted from: https://github.com/xiamengzhou/training_trajectory_analysis/blob/main/utils.py
class CausalLMSubtract(GPT2LMHeadModel):
    def __init__(self, config, model_2, model_1_weight=-1, model_2_weight=1):
        super().__init__(config)
        self.model_2 = GPT2LMHeadModel.from_pretrained(model_2)
        self.model_1_weight = model_1_weight
        self.model_2_weight = model_2_weight

    def forward(self, **kwargs):
        """
        kwargs will include
        - input_ids
        - attention_mask
        - past_key_values: (model 1, model 2)
        - use cache
        - return_dict
        - output_attentions
        - output_hidden_states

        Model 2 should share all of them except past_key_values.
        """
        model_1_input = kwargs.copy()
        model_2_input = kwargs.copy()
        if 'past_key_values' in kwargs and kwargs['past_key_values'] is not None:
            model_1_input['past_key_values'] = kwargs['past_key_values'][0]
            model_2_input['past_key_values'] = kwargs['past_key_values'][1]

        model_1_output = super().forward(**model_1_input)
        model_2_output = self.model_2(**model_2_input)

        subtract_prob = self.model_1_weight * F.softmax(model_1_output.logits, -1) + self.model_2_weight * F.softmax(model_2_output.logits, -1)

        subtract_prob[subtract_prob < 0] = 0
        subtract_prob = subtract_prob + 1e-7
        new_logits = subtract_prob.log() # No need to normalize because this is the logit

        output = CausalLMOutputWithPast(
            loss=(model_1_output.loss, model_2_output.loss),
            logits=new_logits,
            past_key_values=None, # (model_1_output.past_key_values, model_2_output.past_key_values),
            hidden_states=(model_1_output.hidden_states, model_2_output.hidden_states),
            attentions=(model_1_output.attentions, model_2_output.attentions),
        )
        output['model_1_logits'] = model_1_output.logits
        output['model_2_logits'] = model_2_output.logits
        return output



def estimate_cont_divergence(clms, text, n_estimations = 1, generation_length = 20):
    # TODO
    pass
