FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/KazKozDev/footnote-mcp" \
      org.opencontainers.image.description="MCP server for source-grounded web research" \
      org.opencontainers.image.licenses="MIT" \
      io.modelcontextprotocol.server.name="io.github.kazkozdev/footnote-mcp"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# OCR engine for PDF / screenshot text extraction (used by startup_health_check).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (cached layer — only rebuilds when requirements change).
COPY requirements.txt .
RUN pip install -r requirements.txt

# Headless Chromium plus its system libraries (for the browser fetch tier).
RUN python -m playwright install --with-deps chromium

# Application code + console entry point.
COPY . .
RUN pip install .

# MCP speaks over stdio; clients launch this with `docker run -i --rm footnote-mcp`.
ENTRYPOINT ["footnote-mcp"]
