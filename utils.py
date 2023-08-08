import logging

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

class Visualize():
    def plot_samples(data, channel:int, title=None, plot_name="", n_examples =4):

    #     n_rows = int(20 / 5)
    #     plt.figure(figsize=(1* n_rows, 1*n_rows))
    #     if title: plt.suptitle(title)
    #     X, y= data
    #     for idx in range(n_examples):

    #         ax = plt.subplot(n_rows, 5, idx + 1)

    #         image = 255 - X[idx, channel].view((32,32))
    #         ax.imshow(image,cmap='gray')
    #         ax.axis("off")

    #     if plot_name!="":plt.savefig(f"plots/"+plot_name+".png")

    #     plt.tight_layout()
        images, labels = data
        def imshow(img):
            img = img / 2 + 0.5     # unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()

        def show(inp, label):
            fig = plt.gcf()
            plt.imshow(inp.permute(1,2,0))
            plt.title(title)
        # get some random training images

        grid = torchvision.utils.make_grid(images)
        labels = torch.tensor([0,1,0,0])
        show(grid, label=[int(labels[x])for x in range(len(labels))])

        imshow(torchvision.utils.make_grid(images,padding=1))