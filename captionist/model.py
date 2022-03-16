import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.w = nn.Linear(decoder_dim, attention_dim)
        self.u = nn.Linear(encoder_dim, attention_dim)
        self.a = nn.Linear(attention_dim, 1)

    def forward(self, image_features, hidden_state):
        u_hs = self.u(image_features)
        w_ah = self.w(hidden_state)
        combined = torch.tanh(u_hs + w_ah.unsqueeze(1))
        attention_score = self.a(combined)
        attention_score = attention_score.squeeze(2)
        alpha = F.softmax(attention_score, dim=1)
        attention_weights = image_features * alpha.unsqueeze(2)
        attention_weights = attention_weights.sum(dim=1)
        return alpha, attention_weights


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        encoder_cnn = models.resnet50(pretrained=True)
        for parameter in encoder_cnn.parameters():
            parameter.requires_grad_(False)
        modules = list(encoder_cnn.children())[:-2]
        self.encoder_cnn = nn.Sequential(*modules)

    def forward(self, image):
        image_features = self.encoder_cnn(image)
        image_features = image_features.permute(0, 2, 3, 1)
        image_features = image_features.view(image_features.shape[0], -1, image_features.shape[-1])
        return image_features


class Decoder(nn.Module):
    def __init__(self, parameter_dict, device):
        super(Decoder, self).__init__()
        self.vocab_size = parameter_dict['vocab_size']
        self.attention_dim = parameter_dict['attention_dim']
        self.encoder_dim = parameter_dict['encoder_dim']
        self.decoder_dim = parameter_dict['decoder_dim']
        self.embed_size = parameter_dict['embed_size']
        self.drop_rate = parameter_dict['drop_rate']
        self.device = device
        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.attention = Attention(
            encoder_dim=self.encoder_dim,
            decoder_dim=self.decoder_dim,
            attention_dim=self.attention_dim
        )
        self.lstm_cell = nn.LSTMCell(
            input_size=self.embed_size+self.encoder_dim,
            hidden_size=self.decoder_dim,
            bias=True
        )
        self.linear = nn.Linear(self.decoder_dim, self.vocab_size)
        self.dropout = nn.Dropout(self.drop_rate)
        self.initial_h = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.initial_c = nn.Linear(self.encoder_dim, self.decoder_dim)

    def initial_hidden_state(self, encoded_image):
        mean_encoded_image = encoded_image.mean(dim=1)
        init_h = self.initial_h(mean_encoded_image)
        init_c = self.initial_c(mean_encoded_image)
        return init_h, init_c

    def forward(self, encoded_image, captions):
        embeds = self.embedding(captions)
        h, c = self.initial_hidden_state(encoded_image=encoded_image)
        seq_length = len(captions[0])-1
        batch_size = encoded_image.shape[0]
        num_features = encoded_image.size(1)
        predictions = torch.zeros(batch_size, seq_length, self.vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(self.device)
        for i in range(seq_length):
            alpha, context = self.attention(encoded_image, h)
            lstm_input = torch.cat((embeds[:, i], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.linear(self.dropout(h))
            predictions[:, i] = output
            alphas[:, i] = alpha
        return predictions, alpha

    def generate_caption(self, encoded_image, max_length=20, vocab=None):
        batch_size = encoded_image.size(0)
        h, c = self.initial_hidden_state(encoded_image=encoded_image)
        alphas = []
        word = torch.tensor(vocab.stoi['<SOS>']).view(1, -1).to(self.device)
        embeds = self.embedding(word)
        captions = []
        for i in range(max_length):
            alpha, context = self.attention(encoded_image, h)
            alphas.append(alpha.cpu().detach().numpy())
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            output = self.linear(self.dropout(h))
            output = output.view(batch_size, -1)
            predicted_word_idx = output.argmax(dim=1)
            print(predicted_word_idx.item())
            captions.append(predicted_word_idx.item())
            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))
        captions_word = []
        for idx in captions:
            captions_word.append(vocab.itos[idx])
        return captions_word, alphas


class EncoderDecoder(nn.Module):
    def __init__(self, parameter_dict, device):
        super(EncoderDecoder, self).__init__()
        self.image_encoder = ImageEncoder()
        self.decoder = Decoder(
            parameter_dict=parameter_dict,
            device=device
        )

    def forward(self, images, captions):
        encoded_images = self.image_encoder(images)
        outputs = self.decoder(encoded_images, captions)
        return outputs


