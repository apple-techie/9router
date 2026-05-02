import crypto from "crypto";
import { BaseExecutor } from "./base.js";
import { PROVIDERS } from "../config/providers.js";
import { injectReasoningContent } from "../utils/reasoningContentInjector.js";

// Factory CLI version — kept in sync with the latest known build so the
// upstream gateway accepts our requests.
const FACTORY_CLI_VERSION = "0.104.0";
const FACTORY_USER_AGENT = `factory-cli/${FACTORY_CLI_VERSION}`;
const FACTORY_BASE_URL = "https://api.factory.ai";

const PATH = {
  anthropic: "/api/llm/a/v1/messages",
  "openai-responses": "/api/llm/o/v1/responses",
  "openai-chat": "/api/llm/o/v1/chat/completions",
  gemini: "/api/llm/g/v1/generate",
};

// Model → wire family + api_provider header value.
// Catalog mirrors the Factory model registry reverse-engineered in
// kainotomic/omnigate's omnigate-provider-factory crate.
const MODEL_CATALOG = {
  // Claude (Anthropic) — system prompt MUST be hoisted into the first user
  // message; Factory rejects native `system` blocks with 403.
  "claude-opus-4-7":               { family: "anthropic", api_provider: "anthropic" },
  "claude-opus-4-6":               { family: "anthropic", api_provider: "anthropic" },
  "claude-opus-4-6-fast":          { family: "anthropic", api_provider: "anthropic" },
  "claude-sonnet-4-6":             { family: "anthropic", api_provider: "anthropic" },
  "claude-opus-4-5-20251101":      { family: "anthropic", api_provider: "anthropic" },
  "claude-sonnet-4-5-20250929":    { family: "anthropic", api_provider: "anthropic" },
  "claude-haiku-4-5-20251001":     { family: "anthropic", api_provider: "anthropic" },
  // MiniMax M2.7 routes through the Anthropic surface (Fireworks-hosted).
  "minimax-m2.7":                  { family: "anthropic", api_provider: "fireworks" },
  // GPT-5.x — OpenAI Responses surface.
  "gpt-5.5":                       { family: "openai-responses", api_provider: "openai" },
  "gpt-5.5-fast":                  { family: "openai-responses", api_provider: "openai" },
  "gpt-5.4":                       { family: "openai-responses", api_provider: "openai" },
  "gpt-5.4-fast":                  { family: "openai-responses", api_provider: "openai" },
  "gpt-5.4-mini":                  { family: "openai-responses", api_provider: "openai" },
  "gpt-5.3-codex":                 { family: "openai-responses", api_provider: "openai" },
  "gpt-5.3-codex-fast":            { family: "openai-responses", api_provider: "openai" },
  "gpt-5.2":                       { family: "openai-responses", api_provider: "openai" },
  "gpt-5.2-codex":                 { family: "openai-responses", api_provider: "openai" },
  "gpt-5.1-codex-max":             { family: "openai-responses", api_provider: "openai" },
  "gpt-5.1-codex":                 { family: "openai-responses", api_provider: "openai" },
  "gpt-5.1":                       { family: "openai-responses", api_provider: "openai" },
  // Kimi, GLM & MiniMax M2.5 — OpenAI Chat Completions surface (Fireworks-hosted).
  "kimi-k2.5":                     { family: "openai-chat", api_provider: "fireworks" },
  "kimi-k2.6":                     { family: "openai-chat", api_provider: "fireworks" },
  "glm-5.1":                       { family: "openai-chat", api_provider: "fireworks" },
  "glm-5":                         { family: "openai-chat", api_provider: "fireworks" },
  "glm-4.7":                       { family: "openai-chat", api_provider: "fireworks" },
  "minimax-m2.5":                  { family: "openai-chat", api_provider: "fireworks" },
  // Gemini 3.x — Google native surface.
  "gemini-3.1-pro-preview":        { family: "gemini", api_provider: "google" },
  "gemini-3-flash-preview":        { family: "gemini", api_provider: "google" },
};

function lookupModel(model) {
  return MODEL_CATALOG[model] || null;
}

class FactoryUnknownModelError extends Error {
  constructor(model) {
    super(`Unknown Factory model: ${model}. See /v1/models for the supported list.`);
    this.name = "FactoryUnknownModelError";
    this.status = 404;
    this.model = model;
  }
}

// Factory uses "System instructions:\n\n..." as a hoisting prefix so the
// upstream model treats it as the system prompt.
function systemPromptPrefix(text) {
  return `System instructions:\n\n${(text || "").trim()}`;
}

// Anthropic-format hoisting: pull `system` blocks into a leading user message.
function hoistAnthropicSystem(body) {
  const system = body.system;
  if (system === undefined || system === null) return body;

  let blocks = [];
  if (typeof system === "string") {
    const trimmed = system.trim();
    if (trimmed) blocks.push({ type: "text", text: trimmed });
  } else if (Array.isArray(system)) {
    for (const block of system) {
      if (typeof block === "string") {
        const trimmed = block.trim();
        if (trimmed) blocks.push({ type: "text", text: trimmed });
      } else if (block && typeof block === "object") {
        const text = (block.text ?? "").toString().trim();
        if (!text) continue;
        const next = { ...block, type: "text", text };
        blocks.push(next);
      }
    }
  } else if (typeof system === "object" && system.text) {
    const trimmed = system.text.toString().trim();
    if (trimmed) blocks.push({ ...system, type: "text", text: trimmed });
  }

  if (blocks.length === 0) {
    const { system: _, ...rest } = body;
    return rest;
  }

  // Prefix the first block with "System instructions:" so the model still
  // understands the role of the hoisted content.
  const first = blocks[0];
  blocks[0] = { ...first, text: systemPromptPrefix(first.text) };

  const messages = Array.isArray(body.messages) ? [...body.messages] : [];
  messages.unshift({ role: "user", content: blocks });

  const { system: _omitted, ...rest } = body;
  return { ...rest, messages };
}

// OpenAI Chat hoisting: convert leading system/developer messages into a
// single leading user message with the prefix.
function hoistOpenAIChatSystem(body) {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemTexts = [];
  const others = [];

  for (const msg of messages) {
    if (!msg || typeof msg !== "object") continue;
    if (msg.role === "system" || msg.role === "developer") {
      const text = typeof msg.content === "string"
        ? msg.content
        : Array.isArray(msg.content)
          ? msg.content.map(p => p?.text || "").join("")
          : "";
      const trimmed = text.trim();
      if (trimmed) systemTexts.push(trimmed);
      continue;
    }
    others.push(msg);
  }

  if (systemTexts.length === 0) return body;
  const hoisted = { role: "user", content: systemPromptPrefix(systemTexts.join("\n\n")) };
  return { ...body, messages: [hoisted, ...others] };
}

// OpenAI Responses surface fixes: hoist `instructions`, rename `max_tokens`
// → `max_output_tokens`. The shared openai→openai-responses translator at
// open-sse/translator/request/openai-responses.js leaves `max_tokens`
// unchanged, but Factory's /api/llm/o/v1/responses gateway rejects it as an
// unknown parameter (codex's OpenAI backend silently ignores it).
function hoistOpenAIResponses(body) {
  let result = body;

  if (result.max_tokens !== undefined && result.max_output_tokens === undefined) {
    const { max_tokens, ...rest } = result;
    result = { ...rest, max_output_tokens: max_tokens };
  }

  if (typeof result.max_output_tokens === "number" && result.max_output_tokens < 16) {
    result = { ...result, max_output_tokens: 16 };
  }

  const instructions = typeof result.instructions === "string" ? result.instructions.trim() : "";
  if (!instructions) return result;

  let input = result.input;
  if (typeof input === "string") {
    input = [{ type: "message", role: "user", content: [{ type: "input_text", text: input }] }];
  } else if (!Array.isArray(input)) {
    input = [];
  } else {
    input = [...input];
  }

  input.unshift({
    type: "message",
    role: "user",
    content: [{ type: "input_text", text: systemPromptPrefix(instructions) }],
  });

  const { instructions: _omit, ...rest } = result;
  return { ...rest, input };
}

// Strip `id` from Gemini function_call / function_response parts. Our shared
// openai-to-gemini translator emits `id` because antigravity / gemini-cli
// (Google's internal cloudcode-pa surface) accepts it, but Factory's
// /api/llm/g/v1/generate proxies the PUBLIC Gemini API which rejects with
// "Unknown name 'id' at contents[N].parts[M].function_call". Strip on
// continuation tool conversations, keep everything else.
function stripGeminiFunctionIds(body) {
  if (!Array.isArray(body?.contents)) return body;
  let mutated = false;
  const stripIdField = (obj) => {
    if (!obj || !("id" in obj)) return null;
    const { id: _id, ...rest } = obj;
    return rest;
  };
  const contents = body.contents.map(c => {
    if (!Array.isArray(c?.parts)) return c;
    let partsMutated = false;
    const parts = c.parts.map(p => {
      if (!p || typeof p !== "object") return p;
      let next = p;
      for (const key of ["functionCall", "function_call"]) {
        const stripped = stripIdField(next?.[key]);
        if (stripped) { next = { ...next, [key]: stripped }; partsMutated = true; mutated = true; }
      }
      for (const key of ["functionResponse", "function_response"]) {
        const stripped = stripIdField(next?.[key]);
        if (stripped) { next = { ...next, [key]: stripped }; partsMutated = true; mutated = true; }
      }
      return next;
    });
    return partsMutated ? { ...c, parts } : c;
  });
  return mutated ? { ...body, contents } : body;
}

// Gemini hoisting: drop `systemInstruction`, prepend its text as a user content.
function hoistGeminiSystemInstruction(body) {
  const sys = body.systemInstruction;
  if (!sys) return body;

  let text = "";
  if (typeof sys === "string") {
    text = sys;
  } else if (sys.text) {
    text = sys.text;
  } else if (Array.isArray(sys.parts)) {
    text = sys.parts.map(p => p?.text || "").filter(Boolean).join("\n\n");
  }
  text = (text || "").trim();
  if (!text) {
    const { systemInstruction: _, ...rest } = body;
    return rest;
  }

  const contents = Array.isArray(body.contents) ? [...body.contents] : [];
  contents.unshift({ role: "user", parts: [{ text: systemPromptPrefix(text) }] });
  const { systemInstruction: _omit, ...rest } = body;
  return { ...rest, contents };
}

// Factory returns 403 with body like:
//   {"detail":"Forbidden","status":403, ... "(reset after 1m 28s)"}
// for transient rate limits. LiteLLM treats 403 as a permanent failure
// (no retry, no fallback, fires outage alerts). Rewrite these to 429 with
// a Retry-After header so the client does proper backoff.
function parseFactoryResetAfter(text) {
  const m = text.match(/reset after (?:(\d+)m\s*)?(?:(\d+)s)?/i);
  if (!m) return null;
  const minutes = parseInt(m[1] || "0", 10);
  const seconds = parseInt(m[2] || "0", 10);
  const total = minutes * 60 + seconds;
  return total > 0 ? total : null;
}

export class FactoryDroidExecutor extends BaseExecutor {
  constructor() {
    super("factory-droid", PROVIDERS["factory-droid"]);
  }

  async execute(args) {
    // Reject unknown Factory model IDs upstream of the network call. Without
    // this, our lookupModel() used to silently fall back to claude-opus-4-7's
    // family/api_provider while still passing the original (invalid) model ID
    // in the body — Factory then rejected it as 400 "Invalid model ID" and
    // OpenClaw treated that as an upstream failure rather than a config bug.
    if (!lookupModel(args?.model)) {
      throw new FactoryUnknownModelError(args?.model);
    }

    const result = await super.execute(args);
    const r = result?.response;
    if (!r || r.status !== 403) return result;

    // Factory's 403 body is a generic "Forbidden" payload — the rate-limit
    // duration is conveyed via Retry-After / X-RateLimit-Reset-After
    // headers, NOT via "reset after Xs" text in the body.
    const cloned = r.clone();
    const text = await cloned.text();
    let retryAfterSec =
      parseFactoryResetAfter(text)
      || parseInt(r.headers.get("retry-after") || "0", 10)
      || parseInt(r.headers.get("x-ratelimit-reset-after") || "0", 10);

    // Last resort: derive from x-ratelimit-reset (epoch seconds).
    if (!retryAfterSec) {
      const resetEpoch = parseInt(r.headers.get("x-ratelimit-reset") || "0", 10);
      if (resetEpoch > 0) {
        retryAfterSec = Math.max(1, Math.ceil((resetEpoch * 1000 - Date.now()) / 1000));
      }
    }

    // Distinguish a real auth failure (no rate-limit signal anywhere) from
    // a transient throttle. Real auth failures surface as 403; throttles
    // become 429 with Retry-After so OpenClaw treats them as rate_limit
    // (transient → fall back) instead of auth (terminal → surface error).
    if (!retryAfterSec) {
      // Default to 30s when 9router has independent reason to believe this
      // is a Factory throttle (e.g. body lacks "Forbidden" or contains the
      // Factory request-id pattern). Plain 403 with no signal → preserve.
      const looksLikeFactoryThrottle = /requestId":"sfo\d/i.test(text);
      if (!looksLikeFactoryThrottle) return result;
      retryAfterSec = 30;
    }

    const headers = new Headers(r.headers);
    headers.set("Retry-After", String(retryAfterSec));
    headers.set("X-Factory-Original-Status", "403");
    console.log(`[factory-droid] rewriting 403 → 429 with Retry-After=${retryAfterSec}s`);
    return {
      ...result,
      response: new Response(text, { status: 429, statusText: "Too Many Requests", headers }),
    };
  }

  buildUrl(model) {
    this._lastModel = model;
    const entry = lookupModel(model);
    return `${FACTORY_BASE_URL}${PATH[entry.family]}`;
  }

  buildHeaders(credentials, stream = true) {
    const model = this._lastModel;
    const entry = lookupModel(model);
    const apiKey = credentials?.apiKey || credentials?.accessToken;
    const sessionId = crypto.randomUUID();
    const assistantMessageId = `msg_${crypto.randomUUID()}`;

    const headers = {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${apiKey}`,
      "User-Agent": FACTORY_USER_AGENT,
      "x-client-version": FACTORY_CLI_VERSION,
      "x-api-provider": entry.api_provider,
      "x-session-id": sessionId,
      "x-assistant-message-id": assistantMessageId,
    };

    if (entry.family === "anthropic") {
      headers["anthropic-version"] = "2023-06-01";
    }

    if (stream) headers["Accept"] = "text/event-stream";
    return headers;
  }

  transformRequest(model, body) {
    const enriched = injectReasoningContent({ provider: this.provider, model, body });
    const entry = lookupModel(model);
    switch (entry.family) {
      case "anthropic":        return hoistAnthropicSystem(enriched);
      case "openai-responses": return hoistOpenAIResponses(enriched);
      case "openai-chat":      return hoistOpenAIChatSystem(enriched);
      case "gemini":           return hoistGeminiSystemInstruction(stripGeminiFunctionIds(enriched));
      default:                 return enriched;
    }
  }
}

export default FactoryDroidExecutor;
