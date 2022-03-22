from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu


def get_sentence_bleu(references, candidate):
    score = sentence_bleu(references, candidate, weights=(1.0, 0, 0, 0))
    return score


def get_actual_caption(int_list, vocab):
    caption = []
    for i in int_list:
        caption.append(vocab.itos[i.item()])
        if vocab.itos[i.item()] == "<EOS>":
            break
    return caption[1:-1]


def validator(model, data_loader, vocab, device):
    print('Validating................')
    count = 0
    sum_score = 0
    for idx, (images, captions) in enumerate(tqdm(iter(data_loader))):
        for i in range(images.shape[0]):
            features = model.image_encoder(images[i:i+1].to(device))
            caps, _ = model.decoder.generate_caption(features, vocab=vocab)
            actual_caption = get_actual_caption(captions[i], vocab)
            bleu_score = get_sentence_bleu([actual_caption[:-1]], caps[:-2])
            sum_score = sum_score + bleu_score
            count = count + 1
    return sum_score/count
