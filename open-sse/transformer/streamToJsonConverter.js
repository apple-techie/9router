/**
 * Stream-to-JSON Converter
 * Converts Responses API SSE stream to single JSON response
 * Used when client requests non-streaming but provider forces streaming (e.g., Codex)
 */

/**
 * Process a single SSE message and update state accordingly.
 */
function processSSEMessage(msg, state) {
  if (!msg.trim()) return;

  const eventMatch = msg.match(/^event:\s*(.+)$/m);
  const dataMatch = msg.match(/^data:\s*(.+)$/m);
  if (!eventMatch || !dataMatch) return;

  const eventType = eventMatch[1].trim();
  const dataStr = dataMatch[1].trim();
  if (dataStr === "[DONE]") return;

  let parsed;
  try { parsed = JSON.parse(dataStr); }
  catch { return; }

  if (eventType === "response.created") {
    state.responseId = parsed.response?.id || state.responseId;
    state.created = parsed.response?.created_at || state.created;
  } else if (eventType === "response.output_item.done") {
    state.items.set(parsed.output_index ?? 0, parsed.item);
  } else if (eventType === "response.completed") {
    state.status = "completed";
    if (parsed.response?.usage) {
      state.usage.input_tokens = parsed.response.usage.input_tokens || 0;
      state.usage.output_tokens = parsed.response.usage.output_tokens || 0;
      state.usage.total_tokens = parsed.response.usage.total_tokens || 0;
    }
  } else if (eventType === "response.failed") {
    state.status = "failed";
  }
}

const EMPTY_RESPONSE = { input_tokens: 0, output_tokens: 0, total_tokens: 0 };

/**
 * Convert Responses API SSE stream to single JSON response
 * @param {ReadableStream} stream - SSE stream from provider
 * @returns {Promise<Object>} Final JSON response in Responses API format
 */
export async function convertResponsesStreamToJson(stream) {
  if (!stream || typeof stream.getReader !== "function") {
    return { id: `resp_${Date.now()}`, object: "response", created_at: Math.floor(Date.now() / 1000), status: "failed", output: [], usage: { ...EMPTY_RESPONSE } };
  }

  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const state = {
    responseId: "",
    created: Math.floor(Date.now() / 1000),
    status: "in_progress",
    usage: { ...EMPTY_RESPONSE },
    items: new Map()
  };

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const messages = buffer.split("\n\n");
      buffer = messages.pop() || "";

      for (const msg of messages) {
        processSSEMessage(msg, state);
      }
    }

    // Flush remaining buffer (last event may not end with \n\n)
    if (buffer.trim()) {
      processSSEMessage(buffer, state);
    }
  } finally {
    reader.releaseLock();
  }

  // Build output array from accumulated items (ordered by index)
  const output = [];
  const maxIndex = state.items.size > 0 ? Math.max(...state.items.keys()) : -1;
  for (let i = 0; i <= maxIndex; i++) {
    output.push(state.items.get(i) || { type: "message", content: [], role: "assistant" });
  }

  return {
    id: state.responseId || `resp_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    object: "response",
    created_at: state.created,
    status: state.status || "completed",
    output,
    usage: state.usage
  };
}

/**
 * Convert Gemini-native SSE stream to OpenAI chat.completion JSON.
 *
 * Factory's /api/llm/g/v1/generate (and Vertex AI) ALWAYS streams Gemini-native
 * chunks (`{"candidates":[{"content":{"parts":[...]}}], "usageMetadata":...}`)
 * even when the caller asked for non-streaming. The early chunks carry the
 * actual `text`; the terminal chunk carries an empty `text` plus a
 * `thoughtSignature` and `finishReason: "STOP"`.
 *
 * `parseSSEToOpenAIResponse` (the generic SSE→JSON aggregator) only knows the
 * OpenAI delta shape and finds nothing in Gemini chunks → returns empty
 * content. This converter understands the Gemini shape directly.
 *
 * @param {ReadableStream} stream  - SSE response body from Factory/Vertex
 * @param {string} fallbackModel   - Model id to embed in the response
 * @returns {Promise<Object>} OpenAI chat.completion JSON
 */
export async function convertGeminiStreamToOpenAIJson(stream, fallbackModel = "gemini") {
  if (!stream || typeof stream.getReader !== "function") {
    return {
      id: `chatcmpl-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: fallbackModel,
      choices: [{ index: 0, message: { role: "assistant", content: "" }, finish_reason: "stop" }]
    };
  }

  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  let responseId = "";
  let modelVersion = fallbackModel;
  const contentParts = [];
  const reasoningParts = [];
  const toolCallMap = new Map();    // index → { id, type, function: { name, arguments } }
  let toolIndex = 0;
  let finishReason = "stop";
  let usage = null;

  function processGeminiChunk(payload) {
    let parsed;
    try { parsed = JSON.parse(payload); } catch { return; }

    // Antigravity proxy wraps responses in `{response: {...}}`
    const r = parsed.response || parsed;

    if (!responseId && r.responseId) responseId = r.responseId;
    if (r.modelVersion) modelVersion = r.modelVersion;

    const candidate = r.candidates?.[0];
    if (candidate) {
      const parts = candidate.content?.parts || [];
      for (const part of parts) {
        const isThought = part.thought === true;

        if (typeof part.text === "string" && part.text.length > 0) {
          if (isThought) reasoningParts.push(part.text);
          else contentParts.push(part.text);
        }

        if (part.functionCall) {
          const fcName = part.functionCall.name || "";
          const fcArgs = part.functionCall.args || {};
          toolCallMap.set(toolIndex, {
            id: `${fcName}-${Date.now()}-${toolIndex}`,
            type: "function",
            function: { name: fcName, arguments: JSON.stringify(fcArgs) }
          });
          toolIndex++;
        }
      }

      if (candidate.finishReason) {
        finishReason = String(candidate.finishReason).toLowerCase();
        if (finishReason === "stop" && toolCallMap.size > 0) finishReason = "tool_calls";
      }
    }

    if (r.usageMetadata) {
      const u = r.usageMetadata;
      const promptTokens = u.promptTokenCount || 0;
      const candidatesTokens = u.candidatesTokenCount || 0;
      const thoughtsTokens = u.thoughtsTokenCount || 0;
      const completionTokens = candidatesTokens + thoughtsTokens;
      usage = {
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
        total_tokens: promptTokens + completionTokens
      };
      if (thoughtsTokens > 0) {
        usage.completion_tokens_details = { reasoning_tokens: thoughtsTokens };
      }
    }
  }

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (!payload || payload === "[DONE]") continue;
        processGeminiChunk(payload);
      }
    }
    // Flush trailing buffer
    if (buffer.trim().startsWith("data:")) {
      processGeminiChunk(buffer.trim().slice(5).trim());
    }
  } finally {
    reader.releaseLock();
  }

  const message = {
    role: "assistant",
    content: contentParts.join("") || (toolCallMap.size > 0 ? null : "")
  };
  if (reasoningParts.length > 0) message.reasoning_content = reasoningParts.join("");
  if (toolCallMap.size > 0) {
    message.tool_calls = [...toolCallMap.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([, tc]) => tc);
  }

  const result = {
    id: responseId || `chatcmpl-${Date.now()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: modelVersion,
    choices: [{ index: 0, message, finish_reason: finishReason }]
  };
  if (usage) result.usage = usage;
  return result;
}
