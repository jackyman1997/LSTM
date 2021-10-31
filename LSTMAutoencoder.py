import torch


class Encoder(torch.nn.Module):
    def __init__(
        self,
        seq_length,
        n_features,
        output_length,
        hidden_size,
        num_layers: int = 1,
        batch_first: bool = False,
        dropout: float = 0,
        bidirectional: bool = False,
        add_linear: bool = False
    ):
        super().__init__()
        self.seq_length = seq_length
        self.n_features = n_features
        self.output_length = output_length
        self.n_direction = 2 if bidirectional else 1
        self.add_linear = add_linear
        self.LSTM = torch.nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )
        if add_linear:
            self.linear = torch.nn.Linear(
                self.LSTM.hidden_size * self.n_direction,
                self.output_length
            )

    def forward(self, seq):
        # for this case (BATCH_FIRST),
        # seq dim is (batch, seq_length, n_features), check using torch.randn(bitch_size, seq_length, n_features)
        # output is seq, (h_n, c_n) having dim as:
        # (batch, seq_length, n_features*self.D)
        # (batch, self.D*num_layers, hidden_size) since proj_size is set 0 here otherwise it s proj_size
        # (batch, self.D*num_layers, hidden_size)
        if self.add_linear:
            _, (h_n, _) = self.LSTM(X)
            # reshape the h_n (last output) to (num_layer, n_direction, batch=-1 (-1 means all), hidden_size)
            # get the last one, which is the last layer from the h_n
            # then get rid of the num_layer, as we chose the last layer, hence squeeze(dim=0)
            h_n = h_n.reshape(
                self.LSTM.num_layers,
                self.n_direction,
                -1,
                self.LSTM.hidden_size
            )[-1].squeeze(dim=0)
            # if LSTM is bidirectional, concat the 2 direction outputs (last layer of h_n)
            # from above, h_n now has the shape of (n_direction, batch, hidden_size)
            # so h_n[0] is 1 direction, h_n[1] is the other, size of (batch, hidden_size)
            # concat about the hidden_size can keep the batch size no changed
            # hence output the size of (batch, n_direction * hidden_size)
            if self.LSTM.bidirectional:
                h_n = torch.cat((h_n[0], h_n[1]), dim=-1)
            else:
                h_n = h_n.squeeze(dim=0)
            return self.linear(h_n)
        else:
            output, (h_n, c_n) = self.LSTM(X)
            return output, (h_n, c_n)


# below not yet finished
class Decoder(torch.nn.Module):
    def __init__(self, seq_len, input_dim=1, output_dim=377):
        super().__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.output_dim = 2*input_dim, output_dim

        self.rnn = torch.nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            num_layers=1,
            dropout=0
        )

        self.dense_layer = torch.nn.Sequential(
            # torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            # torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1)
        )

        torch.nn.Linear(self.hidden_dim, 1)
        raise NotImplementedError

    def forward(self, x):
        # The input x is fed into the encoder. The code h is then fed into the decoder.

        x = x.repeat(self.seq_len, 1)

        x = x.reshape(
            (1, self.seq_len, self.input_dim)
        )

        x, (hidden_n, cell_n) = self.rnn(x)

        x = x.reshape(
            (self.seq_len, self.hidden_dim)
        )
        x = self.dense_layer(x)
        raise NotImplementedError
        return x.T


class Autoencoder(torch.nn.Module):
    def __init__(self, seq_len=377, output_size=377, embedding_dim=64):
        super().__init__()
        self.seq_len, self.output_size = seq_len, output_size
        self.enbedding_dim = embedding_dim
        # the encoder should have as input size the size of each data sample and as output size the size of the code.
        # FFN using ReLu as activation function.
        self.encoder = Encoder(
            self.seq_len, self.output_size, self.enbedding_dim).to(device)
        # the decoder hould have as input size the size of the code and as output size the size of the each data sample.
        # Another FFN using ReLu as activation function. We want separate neural networks for encoder and decoder.
        self.decoder = Decoder(
            self.seq_len, self.enbedding_dim, self.output_size).to(device)
        raise NotImplementedError

    def forward(self, x):
        # The input x is fed into the encoder. The code h is then fed into the decoder.
        h = self.encoder(x)
        r = self.decoder(h)
        raise NotImplementedError
        return r
