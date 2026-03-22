import torch
import torch.nn as nn
import evaluate
rouge = evaluate.load("rouge")


class next_token_model(nn.Module):
    def __init__(self, tokenizer, max_len, emb_dim=128, hidden_dim=256):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.embedding = nn.Embedding(tokenizer.vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, tokenizer.vocab_size)
        self.init_weights()

    def init_weights(self):
        for name, module in self.named_children():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity='leaky_relu',
                    a=0.01
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
                for param_name, param in module.named_parameters():
                    if 'weight' in param_name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in param_name:
                        nn.init.zeros_(param)
            elif isinstance(module, (nn.Embedding)):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    # def forward(self, input_ids, lengths):
    #     x = self.embedding(input_ids)  # [batch_size, seq_len, hidden_dim]
    #     packed = nn.utils.rnn.pack_padded_sequence(
    #         x, lengths.cpu(), batch_first=True, enforce_sorted=False
    #     )
    #     _, (h_n, _) = self.lstm(packed)  # [batch_size, seq_len, hidden_dim]
    #     x = h_n[-1]
    #     x = self.norm(x)  # [batch_size, hidden_dim]
    #     x = self.dropout(x)
    #     x = self.fc(x)  # [batch_size, vocab_size]
    #     return x
    def forward(self, input_ids, lengths):
        x = self.embedding(input_ids)  # [batch_size, seq_len, hidden_dim]
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )  # [batch_size, seq_len, hidden_dim * num_directions]
        batch_size = output.size(0)
        indices = (lengths - 1).view(-1, 1, 1).expand(batch_size, 1, output.size(2))
        last_outputs = output.gather(1, indices).squeeze(1)  # [batch_size, hidden_dim x num_directions]
        x = self.norm(last_outputs)  # [batch_size, hidden_dim x num_directions]
        x = self.dropout(x)
        x = self.fc(x)  # [batch_size, vocab_size]
        return x
    
    def predict(self, tokens: list):
        device = next(self.parameters()).device.type
        input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
        lengths = torch.tensor([len(tokens)])

        logits = self.forward(input_ids, lengths)
        return logits.argmax(dim=1)

    def generate_tweet_ending(self, tweet, tokens_to_generate=20):
        token_ids = self.tokenizer.encode(tweet)
        for _ in range(tokens_to_generate):
            context = token_ids[-self.max_len:]
            predicted_token = self.predict(context).item()
            token_ids.append(predicted_token)
        tweet_with_ending = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return tweet_with_ending, token_ids

    def complete_tweet(self, tweet):
        tweet_tokenized = self.tokenizer.encode(tweet)
        length = int(len(tweet_tokenized) * 3 / 4)
        tokens_to_generate = int(len(tweet_tokenized) * 1 / 4)
        subtweet = self.tokenizer.decode(tweet_tokenized[:length], skip_special_tokens=True)
        predict, predict_tokens = self.generate_tweet_ending(subtweet, tokens_to_generate)
        return predict, predict_tokens

    def compute_rouges(self, texts):
        references = texts["text"].tolist()

        predictions, refs_to_measure, preds_to_measure = [], [], []
        for tweet in references:
            predict, predict_tokens = self.complete_tweet(tweet)
            predictions.append(predict)
            tweet_tokenized = self.tokenizer.encode(tweet)
            start = int(len(tweet_tokenized) * 3 / 4)
            refs_to_measure.append(self.tokenizer.decode(tweet_tokenized[start:], skip_special_tokens=True))
            preds_to_measure.append(self.tokenizer.decode(predict_tokens[start:], skip_special_tokens=True))

        rouges = rouge.compute(predictions=preds_to_measure, references=refs_to_measure)
        return rouges["rouge1"], rouges["rouge2"]