import torch
import matplotlib.pyplot as plt


def save_model(model, parameters, epoch):
    model_state = {
        'num_epoch': epoch,
        'embed_size': parameters['embed_size'],
        'attention_dim': parameters['attention_dim'],
        'encoder_dim': parameters['encoder_dim'],
        'decoder_dim': parameters['decoder_dim'],
        'state_dict': model.state_dict()
    }
    model_name = f'model_{epoch}.pth'
    torch.save(model_state, model_name)


def show_image(img, title=None):
    img[0] = img[0] * 0.229
    img[1] = img[1] * 0.224
    img[2] = img[2] * 0.225
    img[0] += 0.485
    img[1] += 0.456
    img[2] += 0.406
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def view_image(img, caption):
    plt.figure()
    plt.imshow(img)
    plt.title(caption)
    plt.xticks([])
    plt.yticks([])


def plot_attention(img, result, attention_plot):
    temp_image = img
    fig = plt.figure(figsize=(10, 10))
    len_result = len(result)
    for l in range(len_result):
        temp_att = attention_plot[l].reshape(8, 8)
        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l], fontsize=12)
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.7, extent=img.get_extent())
        ax.set_xticks([], minor=False)
        ax.set_yticks([], minor=False)
    plt.tight_layout()
