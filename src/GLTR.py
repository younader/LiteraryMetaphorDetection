import numpy as np
import torch
import time
import nltk

# from pytorch_pretrained_bert import (GPT2LMHeadModel, GPT2Tokenizer,
#                                      BertTokenizer, BertForMaskedLM)
from transformers import GPT2Tokenizer,GPT2LMHeadModel,GPTNeoForCausalLM

from matplotlib import pyplot as plt

class AbstractLanguageChecker():
    """
    Abstract Class that defines the Backend API of GLTR.

    To extend the GLTR interface, you need to inherit this and
    fill in the defined functions.
    """

    def __init__(self):
        '''
        In the subclass, you need to load all necessary components
        for the other functions.
        Typically, this will comprise a tokenizer and a model.
        '''
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def check_probabilities(self, in_text, topk=40):
        '''
        Function that GLTR interacts with to check the probabilities of words

        Params:
        - in_text: str -- The text that you want to check
        - topk: int -- Your desired truncation of the head of the distribution

        Output:
        - payload: dict -- The wrapper for results in this function, described below

        Payload values
        ==============
        bpe_strings: list of str -- Each individual token in the text
        real_topk: list of tuples -- (ranking, prob) of each token
        pred_topk: list of list of tuple -- (word, prob) for all topk
        '''
        raise NotImplementedError

    def postprocess(self, token):
        """
        clean up the tokens from any special chars and encode
        leading space by UTF-8 code '\u0120', linebreak with UTF-8 code 266 '\u010A'
        :param token:  str -- raw token text
        :return: str -- cleaned and re-encoded token text
        """
        raise NotImplementedError


def top_k_logits(logits, k):
    '''
    Filters logits to only the top k choices
    from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py
    '''
    if k == 0:
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1]
    return torch.where(logits < min_values,
                       torch.ones_like(logits, dtype=logits.dtype) * -1e10,
                       logits)



class LM(AbstractLanguageChecker):
    def __init__(self, model_name_or_path="gpt2"):
        super(LM, self).__init__()
        self.enc = GPT2Tokenizer.from_pretrained(model_name_or_path)
        if "neo" in model_name_or_path.lower():
          self.model=GPTNeoForCausalLM.from_pretrained(model_name_or_path)
        else:
          self.model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        self.start_token = '<|endoftext|>'
        print("Loaded GPT-2 model!")

    def check_probabilities(self, in_text, topk=40):
        # Process input
        start_t = torch.full((1, 1),
                             self.enc.encoder[self.start_token],
                             device=self.device,
                             dtype=torch.long)
        context = self.enc.encode(in_text)
        context = torch.tensor(context,
                               device=self.device,
                               dtype=torch.long).unsqueeze(0)
        context = torch.cat([start_t, context], dim=1)
        # Forward through the model
        # logits, _ = self.model(context)
        output=self.model(context)
        logits=output[0]
        # construct target and pred
        yhat = torch.softmax(logits[0, :-1], dim=-1)
        y = context[0, 1:]
        # Sort the predictions for each timestep
        sorted_preds = np.argsort(-yhat.data.cpu().numpy())
        # [(pos, prob), ...]
        real_topk_pos = list(
            [int(np.where(sorted_preds[i] == y[i].item())[0][0])
             for i in range(y.shape[0])])
        real_topk_probs = yhat[np.arange(
            0, y.shape[0], 1), y].data.cpu().numpy().tolist()
        real_topk_probs = list(map(lambda x: round(x, 5), real_topk_probs))

        real_topk = list(zip(real_topk_pos, real_topk_probs))
        # [str, str, ...]
        bpe_strings = [self.enc.decoder[s.item()] for s in context[0]]

        bpe_strings = [self.postprocess(s) for s in bpe_strings]

        # [[(pos, prob), ...], [(pos, prob), ..], ...]
        pred_topk = [
            list(zip([self.enc.decoder[p] for p in sorted_preds[i][:topk]],
                     list(map(lambda x: round(x, 5),
                              yhat[i][sorted_preds[i][
                                      :topk]].data.cpu().numpy().tolist()))))
            for i in range(y.shape[0])]

        pred_topk = [[(self.postprocess(t[0]), t[1]) for t in pred] for pred in pred_topk]
        payload = {'bpe_strings': bpe_strings,
                   'real_topk': real_topk,
                   'pred_topk': pred_topk}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return payload

    def sample_unconditional(self, length=100, topk=5, temperature=1.0):
        '''
        Sample `length` words from the model.
        Code strongly inspired by
        https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_gpt2.py

        '''
        context = torch.full((1, 1),
                             self.enc.encoder[self.start_token],
                             device=self.device,
                             dtype=torch.long)
        prev = context
        output = context
        past = None
        # Forward through the model
        with torch.no_grad():
            for i in range(length):
                logits, past = self.model(prev, past=past)
                logits = logits[:, -1, :] / temperature
                # Filter predictions to topk and softmax
                probs = torch.softmax(top_k_logits(logits, k=topk),
                                      dim=-1)
                # Sample
                prev = torch.multinomial(probs, num_samples=1)
                # Construct output
                output = torch.cat((output, prev), dim=1)

        output_text = self.enc.decode(output[0].tolist())
        return output_text

    def postprocess(self, token):
        with_space = False
        with_break = False
        if token.startswith('Ġ'):
            with_space = True
            token = token[1:]
            # print(token)
        elif token.startswith('â'):
            token = ' '
        elif token.startswith('Ċ'):
            token = ' '
            with_break = True

        token = '-' if token.startswith('â') else token
        token = '“' if token.startswith('ľ') else token
        token = '”' if token.startswith('Ŀ') else token
        token = "'" if token.startswith('Ļ') else token

        if with_space:
            token = '\u0120' + token
        if with_break:
            token = '\u010A' + token

        return token