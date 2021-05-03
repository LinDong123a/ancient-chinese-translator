import argparse


def get_args_by_parser(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> dict:
    return {
        a.dest: getattr(args, a.dest, None) for a in parser._group_actions
    }
