# Claude Code Gateway

> Turn your Claude Code subscription into an API endpoint for any AI agent -- no separate Anthropic API key needed. Works with OpenClaw, ZeroClaw, Cline, Aider, Open WebUI, LangChain, and 30+ tools.

## What Is Claude Code Gateway?

Claude Code Gateway is a free, open-source Python server that wraps your local Claude Code CLI as an OpenAI-compatible API. It exposes a standard `/v1/chat/completions` endpoint so any tool that speaks the OpenAI format -- Cline, Aider, LangChain, Open WebUI, and dozens more -- can use Claude models directly through your existing Claude Code subscription. There is no per-token billing and no separate Anthropic API key to purchase.

## Why Use Claude Code Gateway?

AI coding tools like Cline, Aider, Continue.dev, and Open WebUI require an OpenAI-compatible API endpoint to function. The standard Anthropic API charges per token on top of your subscription, which adds up fast during heavy development sessions.

Claude Code Gateway solves this by routing all requests through the Claude Code CLI included with your Claude Max or Pro subscription. You get the same Claude Sonnet, Opus, and Haiku models powering your favorite tools at no additional API cost. The gateway is a lightweight FastAPI server that translates OpenAI-format requests into Claude Code CLI calls and streams the responses back.

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/enescingoz/claude-code-gateway.git
cd claude-code-gateway
cp .env.example .env
docker compose up --build
```

### Local

```bash
git clone https://github.com/enescingoz/claude-code-gateway.git
cd claude-code-gateway
pip install -r requirements.txt
cp .env.example .env
uvicorn src.main:app --host 0.0.0.0 --port 8080
```

Point your tool's OpenAI base URL to `http://localhost:8080/v1` and you're done.

## Compatible Tools

Claude Code Gateway works with any tool that supports an OpenAI-compatible API endpoint. The gateway accepts both native Claude model names (`claude-sonnet-4-20250514`) and GPT model names (`gpt-4o`), automatically mapping them to the correct Claude model.

### Coding Agents

| Tool | Description |
|------|-------------|
| [Cline](https://github.com/cline/cline) | Autonomous coding agent for VS Code |
| [Aider](https://github.com/Aider-AI/aider) | Terminal-based AI pair programmer |
| [OpenHands](https://github.com/OpenHands/OpenHands) | Autonomous software development agent |
| [bolt.diy](https://github.com/stackblitz-labs/bolt.diy) | Browser-based full-stack app builder |
| [Goose](https://github.com/block/goose) | On-machine coding agent by Block |
| [Roo Code](https://github.com/RooVetGit/Roo-Code) | AI dev team inside VS Code |
| [SWE-agent](https://github.com/SWE-agent/SWE-agent) | Autonomous GitHub issue resolver |
| [OpenAI Codex CLI](https://github.com/openai/codex) | Lightweight terminal coding agent |

### Chat Interfaces

| Tool | Description |
|------|-------------|
| [Open WebUI](https://github.com/open-webui/open-webui) | Self-hosted ChatGPT-like interface |
| [LibreChat](https://github.com/danny-avila/LibreChat) | Multi-provider chat with agents and RAG |
| [Lobe Chat](https://github.com/lobehub/lobe-chat) | Extensible multi-provider chat framework |
| [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) | All-in-one AI app with RAG and agents |
| [Jan](https://github.com/janhq/jan) | Offline-first desktop AI app |
| [ChatBot UI](https://github.com/mckaywrigley/chatbot-ui) | Minimal open-source chat interface |

### Agent Frameworks

| Tool | Description |
|------|-------------|
| [LangChain](https://github.com/langchain-ai/langchain) | Most popular agent orchestration library |
| [CrewAI](https://github.com/crewAIInc/crewAI) | Role-based multi-agent framework |
| [AutoGen](https://github.com/microsoft/autogen) | Microsoft's multi-agent framework |
| [LlamaIndex](https://github.com/run-llama/llama_index) | RAG and data-augmented agent framework |
| [Pydantic AI](https://github.com/pydantic/pydantic-ai) | Type-safe agent framework |
| [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) | Official multi-agent SDK |
| [Smolagents](https://github.com/huggingface/smolagents) | HuggingFace's minimal agent framework |

### IDE Extensions

| Tool | Description |
|------|-------------|
| [Continue.dev](https://github.com/continuedev/continue) | AI assistant for VS Code and JetBrains |
| [Tabby](https://github.com/TabbyML/tabby) | Self-hosted code completion server |
| [Void](https://github.com/voideditor/void) | Open-source AI code editor |
| [Zed](https://github.com/zed-industries/zed) | High-performance editor with AI assistant |

### Visual Builders

| Tool | Description |
|------|-------------|
| [n8n](https://github.com/n8n-io/n8n) | Visual workflow automation |
| [Dify](https://github.com/langgenius/dify) | LLMOps platform with visual workflows |
| [Flowise](https://github.com/FlowiseAI/Flowise) | Drag-and-drop agent builder |
| [Langflow](https://github.com/langflow-ai/langflow) | Low-code visual agent builder |

## Tool Setup Guides

Each tool below connects to Claude Code Gateway using the OpenAI-compatible `/v1/chat/completions` endpoint at `http://localhost:8080/v1`. Start the gateway first using the Quick Start instructions above, then configure your tool.

### Using with Cline

In Cline settings, set the API Provider to "OpenAI Compatible". Set the Base URL to `http://localhost:8080/v1`. Enter any string as the API key (for example, `not-needed`). Select your preferred Claude model. No Anthropic API key is required.

### Using with Aider

Run Aider with the OpenAI base URL flag pointing to your local gateway:

```bash
aider --openai-api-base http://localhost:8080/v1 --openai-api-key not-needed
```

### Using with Open WebUI

In Open WebUI admin settings, add a new OpenAI-compatible connection. Set the base URL to `http://localhost:8080/v1` and enter any value as the API key. Claude models will appear in the model selector.

### Using with Continue.dev

In your `.continue/config.json` file, add a provider entry with the gateway as the base URL:

```json
{
  "models": [{
    "provider": "openai",
    "title": "Claude via Gateway",
    "apiBase": "http://localhost:8080/v1",
    "apiKey": "not-needed",
    "model": "claude-sonnet-4-20250514"
  }]
}
```

### Using with LangChain

Use the `ChatOpenAI` class from `langchain_openai` and point it at the gateway:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed",
    model="sonnet",
)

response = llm.invoke("Explain how dependency injection works in Python")
print(response.content)
```

## How It Works

Claude Code Gateway is a Python FastAPI server that accepts OpenAI-format HTTP requests and translates them into Claude Code CLI invocations. When a request arrives at `/v1/chat/completions`, the gateway resolves the model name (mapping GPT names like `gpt-4o` to Claude equivalents like `claude-sonnet-4-20250514`), formats the message history into a CLI prompt, and invokes the Claude Code CLI as a subprocess. For streaming requests with `stream: true`, the gateway converts Claude's NDJSON output into standard SSE (Server-Sent Events) chunks that match the OpenAI streaming format. Session continuity is maintained across multi-turn conversations. The `/v1/models` endpoint lists available Claude models in OpenAI-compatible format.

## Comparison with Alternatives

Claude Code Gateway is one of several projects that bridge Claude models to OpenAI-compatible tooling. The key difference is that it uses the Claude Code CLI from your existing subscription, avoiding per-token API charges entirely.

| Project | How It Works | What You Need | Streaming |
|---------|-------------|---------------|-----------|
| **claude-code-gateway** (this) | Wraps Claude Code CLI as OpenAI API | Claude Code subscription | Yes |
| claude-code-proxy | Translates Claude API to OpenAI format | Anthropic API key | Yes |
| ccproxy | Claude Max to Cursor proxy | Claude Max subscription | Partial |
| LiteLLM | Multi-provider API router | API keys per provider | Yes |

## Configuration

Copy `.env.example` to `.env` and adjust values as needed. All settings use the `CCG_` prefix.

| Variable | Default | Description |
|----------|---------|-------------|
| CCG_HOST | 0.0.0.0 | Server bind address |
| CCG_PORT | 8080 | Server port |
| CCG_DEFAULT_MODEL | claude-sonnet-4-20250514 | Default Claude model |
| CCG_DEFAULT_MAX_TURNS | 10 | Max conversation turns |
| CCG_CLAUDE_CLI_TIMEOUT | 300 | CLI timeout in seconds |
| CCG_WORKING_DIR | (empty) | Working directory for CLI |

## API Endpoints

Claude Code Gateway exposes three HTTP endpoints. All endpoints follow the OpenAI API format, so any client library or tool designed for the OpenAI API works without modification.

| Method | Path | Description |
|--------|------|-------------|
| POST | /v1/chat/completions | Chat completions (streaming + non-streaming) |
| GET | /v1/models | List available models |
| GET | /health | Health check |

## FAQ

### Do I need an Anthropic API key?

No. Claude Code Gateway uses the Claude Code CLI, which authenticates through your existing Claude subscription (Max or Pro). There is no separate API key to purchase and no per-token billing.

### Which Claude models are supported?

Claude Sonnet 4 (`claude-sonnet-4-20250514`), Claude Opus 4 (`claude-opus-4-20250514`), and Claude Haiku 4.5 (`claude-haiku-4-5-20251001`). You can also use GPT model names like `gpt-4o` or `gpt-3.5-turbo` -- the gateway automatically maps them to the corresponding Claude model.

### Is this free?

The gateway itself is free and open-source under the MIT license. You need a Claude Code subscription (included with Claude Max or Claude Pro) to authenticate the underlying CLI.

### How is this different from the Anthropic API?

The Anthropic API charges per token for every request. Claude Code Gateway routes requests through the Claude Code CLI, which is included in your Claude Max or Pro subscription at no additional per-token cost.

### Does it support streaming?

Yes. Claude Code Gateway supports full SSE streaming via the `/v1/chat/completions` endpoint with `stream: true`. Streaming responses are delivered as standard Server-Sent Events in the OpenAI chunk format, compatible with all major client libraries.

### Can I use GPT model names?

Yes. The gateway maps GPT model names to Claude equivalents automatically. `gpt-4o` and `gpt-4` map to Claude Sonnet 4, `gpt-3.5-turbo` and `gpt-4o-mini` map to Claude Haiku 4.5. You can also use shorthand names like `sonnet`, `opus`, and `haiku`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on submitting issues and pull requests.

## License

MIT -- see [LICENSE](LICENSE) for details.
