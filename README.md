---
title: FluxFoundry
emoji: 🛠️
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: true
tags:
- mcp-server-track
- modal
short_description: Create Flux Finetunes with Modal + MCP
---

Create LoRA finetunes of the Flux image model with Modal + MCP. One click install!

> 🎥 **[Watch the Demo Video](https://www.loom.com/share/ed054eb997024730b129d8d7f48981d9)** - See FluxFoundry in action!

## Installation
[![Install MCP Server](https://cursor.com/deeplink/mcp-install-dark.svg)](https://cursor.com/install-mcp?name=FluxFoundry&config=eyJ1cmwiOiJodHRwczovL2FnZW50cy1tY3AtaGFja2F0aG9uLWZsdXhmb3VuZHJ5LmhmLnNwYWNlL2dyYWRpb19hcGkvbWNwL3NzZSJ9) if you are on Cursor 1.0+

Otherwise install like this
```
{
  "mcpServers": {
    "FluxFoundry": {
      "url": "https://agents-mcp-hackathon-fluxfoundry.hf.space/gradio_api/mcp/sse"
    }
  }
}
```

Optionally install [SecretButler](https://github.com/stillerman/secret-butler) for easy key management.