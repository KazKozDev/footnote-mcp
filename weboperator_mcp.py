"""Console entry point for the WebOperator MCP server.

The runtime is a set of flat top-level modules (server, core, search, ...). This
uniquely-named launcher prepends its own directory to sys.path so those modules
win over any same-named modules elsewhere on the path (avoids shadowing when other
projects ship a generic ``server``/``core``/``search`` module).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cli():
    from server import cli as _cli

    _cli()


if __name__ == "__main__":
    cli()
