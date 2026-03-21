from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import evaluate

rouge = evaluate.load("rouge")


class transformer_generator():
    def __init__(self, model_name):
        self.model_name=model_name
        self.model=AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        self.pipeline=pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1
        )

    def slice_tweet(self, tweet):
        tweet_tokenized = self.tokenizer.encode(tweet)
        length = int(len(tweet_tokenized))
        start = int(length*3/4)
        tweet_start = self.tokenizer.decode(tweet_tokenized[:start], skip_special_tokens=True)
        tweet_end = self.tokenizer.decode(tweet_tokenized[start:], skip_special_tokens=True)
        return tweet_start, tweet_end, tweet_tokenized

    def complete_tweet(self, tweet=None, tweet_start=None, tweet_tokenized=None):
        if tweet_start is None or tweet_tokenized is None:
            tweet_start, _, tweet_tokenized = self.slice_tweet(tweet)
        predict = tweet or tweet_start
        if len(tweet_tokenized) > 1:
            out = self.pipeline(
                tweet_start,
                max_length=len(tweet_tokenized),
                truncation=True,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                top_p=0.95,
                temperature=0.8
            )
            predict = out[0]["generated_text"]
        return predict
    
    def compute_rouges(self, texts):
        references = texts["text"].tolist()
        refs_to_measure, preds_to_measure = [], []
        for tweet in tqdm(references):
            tweet_start, _, tweet_tokenized = self.slice_tweet(tweet)
            predict = self.complete_tweet(tweet_start=tweet_start, tweet_tokenized=tweet_tokenized)
            refs_to_measure.append(tweet[len(tweet_start):])
            preds_to_measure.append(predict[len(tweet_start):])

        rouges = rouge.compute(predictions=preds_to_measure, references=refs_to_measure)
        return rouges["rouge1"], rouges["rouge2"]
