"""footnote-mcp: an MCP server for source-grounded web research.

The runtime lives as submodules of this package (``server``, ``core``,
``search``, ...). ``cli`` is the console-script entry point; the import of the
heavy ``server`` module is deferred into the call so ``import footnote_mcp`` stays
cheap.
"""


def cli():
    from .server import cli as _cli

    _cli()
