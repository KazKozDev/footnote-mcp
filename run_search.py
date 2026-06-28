#!/usr/bin/env python3
import sys
sys.path.insert(0, "/Users/artemk/projects/weboperator-mcp")

from tools_search import web_search

result = web_search("новости Пензы", lang="ru", num=10)
import json
print(json.dumps(result, ensure_ascii=False, indent=2))
