from argparse import ArgumentParser

from tool import Tool


def parse_args():
    """
    Parses arguments for the inference
    """
    parser = ArgumentParser()
    parser.add_argument(
        "-path",
        "--output_path",
        dest="path",
        required=True,
        help="path to save the experiment",
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint_path",
        dest="ckpt",
        required=False,
        default="./pretrained_models/best_model.pt",
        help="path to the pretrained model",
    )
    parser.add_argument(
        "-src",
        "--source",
        dest="source",
        required=True,
        help="name of the source/motion image file",
    )
    parser.add_argument(
        "-trg",
        "--target",
        dest="target",
        required=False,
        help="name of the identity image file",
    )
    parser.add_argument(
        "-lat",
        "--latent",
        dest="latent",
        required=False,
        help="name of the identity latent file",
    )
    parser.add_argument(
        "-rnd",
        "--random_identity",
        dest="random_identity",
        required=False,
        default=False,
        type=bool,
        help="name of the identity latent file",
    )
    parser.add_argument(
        "-algn",
        "--aligned",
        dest="aligned",
        required=False,
        default=False,
        type=bool,
        help="name of the identity latent file",
    )
    parser.add_argument(
        "-cpl",
        "--coupled",
        dest="coupled",
        required=False,
        default=False,
        type=bool,
        help="Add original motion and identity images to the output",
    )
    args = vars(parser.parse_args())
    check_args(args)
    return args


def check_args(args):
    """
    Checks if identity source is present
    """
    print(args)
    assert (
        args["target"] is not None
        or args["latent"] is not None
        or args["random_identity"] is True
    ), "You need to set either target identity --latent for the latent file, --target for the image file or --random_identity=True for random idenitity"


if __name__ == "__main__":
    args = parse_args()
    tool = Tool(opts=None, result_path=args["path"], checkpoint_path=args["ckpt"])
    tool.run(
        args["source"],
        args["target"],
        args["latent"],
        args["random_identity"],
        args["aligned"],
        args["coupled"],
    )
    print("All done")
