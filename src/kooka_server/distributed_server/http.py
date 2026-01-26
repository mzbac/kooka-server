from __future__ import annotations

import json
import logging
import os
import select
import socket
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Empty, Queue
from socketserver import ThreadingMixIn
from typing import Any, List, Optional

from ..api.anthropic.messages import (
    convert_anthropic_to_openai_messages,
    convert_anthropic_tools,
    process_message_content,
)
from ..api.models_endpoint import list_models as list_v1_models
from ..api.openai.tool_calls import make_openai_tool_call, normalize_finish_reason_for_tool_calls
from ..logging_utils import redact_request_body
from ..tool_fixes import (
    ToolFixContext,
    apply as apply_tool_fixes,
    infer_tool_parser_type,
)
from .constants import (
    DEFAULT_REPETITION_CONTEXT_SIZE,
    DEFAULT_REPETITION_PENALTY,
    MAX_STOP_SEQUENCES,
    MAX_STOP_SEQUENCE_LENGTH,
)


def apply_chat_template_safe(
    tokenizer: Any,
    messages: list[dict],
    *,
    tools: Any,
    add_generation_prompt: bool,
    tokenize: bool,
) -> str:
    """Apply a chat template while normalizing empty tools to None."""
    return tokenizer.apply_chat_template(
        messages,
        tools=tools or None,
        add_generation_prompt=add_generation_prompt,
        tokenize=tokenize,
    )


class BadRequestError(Exception):
    pass


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Thread-per-request HTTP server."""
    daemon_threads = True


class DistributedHandler(BaseHTTPRequestHandler):
    """HTTP request handler for distributed inference."""

    def __init__(self, dist_state, tokenizer, args, *handler_args, **handler_kwargs):
        self.dist_state = dist_state
        self.tokenizer = tokenizer
        self.args = args
        self.created = int(time.time())
        super().__init__(*handler_args, **handler_kwargs)

    def log_message(self, format, *args):
        logging.debug(f"[HTTP] {format % args}")

    def _set_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {"status": "ok"})
        elif self.path.startswith("/v1/models"):
            self._handle_models_request()
        else:
            self.send_error(404)

    def _handle_models_request(self):
        models = list_v1_models(created=self.created, active_model=self.args.model, request_path=self.path)
        self._json_response(200, {"object": "list", "data": models})

    def do_POST(self):
        path = self.path.split("?")[0]  # Remove query string

        handlers = {
            "/v1/chat/completions": self._handle_chat,
            "/chat/completions": self._handle_chat,
            "/v1/completions": self._handle_text,
            "/v1/messages": self._handle_anthropic,
        }

        handler = handlers.get(path)
        if handler is None:
            self.send_error(404)
            return

        try:
            handler()
        except BadRequestError as e:
            self._json_response(400, {"error": str(e)})
        except Exception as e:
            logging.exception("Request error")
            self._json_response(500, {"error": str(e)})

    def _parse_body(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode())
        except json.JSONDecodeError as e:
            raise BadRequestError(f"Invalid JSON in request body: {e}") from e
        if not isinstance(body, dict):
            raise BadRequestError("Request body must be a JSON object")
        return body

    def _json_response(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _stream_response(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # Disable proxy buffering for SSE (e.g. nginx) so keepalive bytes are forwarded immediately.
        self.send_header("X-Accel-Buffering", "no")
        self._set_cors_headers()
        self.end_headers()

    def _parse_seed(self, body: dict) -> tuple[Optional[int], bool]:
        seed = body.get("seed", None)
        seed_is_user = seed is not None
        if seed is None:
            return None, False
        try:
            return int(seed), seed_is_user
        except (TypeError, ValueError):
            return None, False

    def _parse_max_tokens(self, body: dict) -> int:
        max_tokens = body.get("max_completion_tokens", None)
        if max_tokens is None:
            max_tokens = body.get("max_tokens", self.args.max_tokens)
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError) as e:
            raise BadRequestError("max_tokens must be a non-negative integer") from e
        if max_tokens < 0:
            raise BadRequestError("max_tokens must be a non-negative integer")
        return max_tokens

    def _parse_sampling(self, body: dict) -> tuple[float, float, int, float, int]:
        temperature = body.get("temperature", self.args.temperature)
        top_p = body.get("top_p", body.get("topP", self.args.top_p))
        top_k = body.get("top_k", body.get("topK", self.args.top_k))
        repetition_penalty = body.get("repetition_penalty", DEFAULT_REPETITION_PENALTY)
        repetition_context_size = body.get(
            "repetition_context_size", DEFAULT_REPETITION_CONTEXT_SIZE
        )

        try:
            temperature = float(temperature)
        except (TypeError, ValueError):
            temperature = float(self.args.temperature)
        try:
            top_p = float(top_p)
        except (TypeError, ValueError):
            top_p = float(self.args.top_p)
        try:
            top_k = int(top_k)
        except (TypeError, ValueError):
            top_k = int(self.args.top_k)
        try:
            repetition_penalty = float(repetition_penalty)
        except (TypeError, ValueError):
            repetition_penalty = DEFAULT_REPETITION_PENALTY
        try:
            repetition_context_size = int(repetition_context_size)
        except (TypeError, ValueError):
            repetition_context_size = DEFAULT_REPETITION_CONTEXT_SIZE

        if temperature < 0:
            temperature = float(self.args.temperature)
        if top_p < 0 or top_p > 1:
            top_p = float(self.args.top_p)
        if top_k < 0:
            top_k = int(self.args.top_k)
        if repetition_penalty < 0:
            repetition_penalty = DEFAULT_REPETITION_PENALTY
        if repetition_context_size < 0:
            repetition_context_size = DEFAULT_REPETITION_CONTEXT_SIZE

        return temperature, top_p, top_k, repetition_penalty, repetition_context_size

    def _parse_stop_token_sequences(self, stop_words: object) -> List[List[int]]:
        if isinstance(stop_words, str):
            stop_words = [stop_words]
        if not isinstance(stop_words, list):
            stop_words = []
        stop_words = [s for s in stop_words if isinstance(s, str) and s]

        stop_token_sequences: List[List[int]] = []
        for sw in stop_words[:MAX_STOP_SEQUENCES]:
            try:
                seq = self.tokenizer.encode(sw, add_special_tokens=False)
            except TypeError:
                seq = self.tokenizer.encode(sw)
            if seq:
                stop_token_sequences.append(seq)
        return stop_token_sequences

    def _handle_chat(self):
        body = self._parse_body()
        messages = body.get("messages", [])
        tools = body.get("tools")
        tool_choice = body.get("tool_choice")
        stream = body.get("stream", False)
        stream_options = body.get("stream_options", None)
        max_tokens = self._parse_max_tokens(body)
        temperature, top_p, top_k, repetition_penalty, repetition_context_size = self._parse_sampling(body)
        seed, seed_is_user = self._parse_seed(body)
        stop_token_sequences = self._parse_stop_token_sequences(body.get("stop") or [])
        model = body.get("model", self.args.model)

        # Tool calling is particularly sensitive to sampling randomness, and
        # opencode-style clients expect a reliable tool_calls finish when
        # tool_choice is forced. If the user didn't explicitly provide sampling
        # parameters, use greedy decoding for forced tool_choice requests.
        forced_tool_choice = isinstance(tool_choice, dict) or tool_choice == "required"
        if tools and forced_tool_choice:
            if "temperature" not in body:
                temperature = 0.0
            if "top_p" not in body and "topP" not in body:
                top_p = 1.0
            if "top_k" not in body and "topK" not in body:
                top_k = 0

        logging.info(f"Received request: model={model}, max_tokens={max_tokens}, stream={stream}, num_messages={len(messages)}")
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            try:
                debug_obj = redact_request_body(body)
                debug_obj["stream_options"] = stream_options
                logging.debug("Request body (redacted): %s", json.dumps(debug_obj, ensure_ascii=False))
            except Exception:
                logging.debug("Failed to render redacted request body", exc_info=True)

        # Map Claude model names to local
        if model.startswith(("claude-", "anthropic")):
            model = self.args.model

        process_message_content(messages)

        prompt = apply_chat_template_safe(
            self.tokenizer,
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )
        emit_initial_think = prompt.rstrip().endswith("<think>")
        prompt_tokens = self.tokenizer.encode(prompt)
        
        logging.info(f"Processing prompt: {len(prompt_tokens)} tokens")

        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        response_queue = Queue()
        self.dist_state.request_queue.put({
            "request_id": request_id,
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens,
            "seed": seed,
            "seed_is_user": seed_is_user,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "repetition_context_size": repetition_context_size,
            "stop_token_sequences": stop_token_sequences,
            "response_queue": response_queue,
            "tools": tools,
        })

        if stream:
            self._stream_chat(response_queue, request_id, model, tools, stream_options, emit_initial_think)
        else:
            self._blocking_chat(response_queue, request_id, model, tools, emit_initial_think)

    def _stream_chat(self, queue, request_id, model, tools, stream_options, emit_initial_think: bool = False):
        self._stream_response()

        has_tool_calling = getattr(self.tokenizer, "has_tool_calling", False)
        tool_call_start = getattr(self.tokenizer, "tool_call_start", None)
        tool_call_end = getattr(self.tokenizer, "tool_call_end", None)
        tool_parser = getattr(self.tokenizer, "tool_parser", None)

        in_tool_call = False
        tool_calls: List[str] = []
        tool_text = ""

        content_buffer = ""
        saw_tool_calls = False
        finish_reason = None
        prompt_toks = 0
        gen_toks = 0
        tool_idx = 0

        tool_parser_type = infer_tool_parser_type(self.tokenizer)
        tool_fix_ctx = ToolFixContext(
            tool_parser_type=tool_parser_type,
            tools=tools,
        )

        def parse_single_tool(tool_text: str) -> Optional[dict]:
            nonlocal tool_idx
            if tool_parser is None:
                logging.warning(
                    "Tool call emitted but tokenizer has no tool_parser (id=%s)",
                    request_id,
                )
                return None
            if not tool_text or not tool_text.strip():
                logging.warning("Skipping empty tool call text (id=%s)", request_id)
                return None
            try:
                tool_call = tool_parser(tool_text, tools)
            except Exception as e:
                preview = tool_text.strip().replace("\n", "\\n")
                if len(preview) > 400:
                    preview = preview[:400] + "…"
                logging.warning(
                    "Failed to parse tool call (id=%s): %s; preview=%r",
                    request_id,
                    e,
                    preview,
                )
                return None
            if not isinstance(tool_call, dict):
                logging.warning(
                    "Tool parser returned non-dict (id=%s): %s",
                    request_id,
                    type(tool_call).__name__,
                )
                return None

            tool_call = apply_tool_fixes(tool_call, tool_fix_ctx)
            arguments = tool_call.get("arguments", {})
            tool_call["arguments"] = json.dumps(arguments, ensure_ascii=False)
            out = make_openai_tool_call(
                name=str(tool_call.get("name") or ""),
                arguments=tool_call.get("arguments") or "{}",
                tool_call_id=str(uuid.uuid4()),
                index=tool_idx,
            )
            tool_idx += 1
            return out

        def parse_tools(tool_calls: List[str]) -> List[dict]:
            if not tool_calls:
                return []
            parsed = []
            for tool_text in tool_calls:
                tc = parse_single_tool(tool_text)
                if tc is not None:
                    parsed.append(tc)
            return parsed

        def send_chunk(
            content: str,
            tool_call_texts: Optional[List[str]] = None,
            finish: Optional[str] = None,
            force: bool = False,
        ):
            nonlocal saw_tool_calls
            content_to_send = content if content else ""
            chunk_tool_calls = parse_tools(tool_call_texts or [])

            if chunk_tool_calls:
                saw_tool_calls = True

            if finish == "tool_calls" and not saw_tool_calls:
                finish = "stop"

            if not force and not content_to_send and not chunk_tool_calls:
                return

            delta = {
                "role": "assistant",
                "content": content_to_send,
                "tool_calls": chunk_tool_calls,
            }

            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()

        # Emit an initial chunk to avoid idle timeouts during long prefill.
        try:
            send_chunk("", force=True)
        except Exception:
            pass

        def client_disconnected() -> bool:
            try:
                conn = getattr(self, "connection", None)
                if conn is None:
                    return False
                readable, _, _ = select.select([conn], [], [], 0)
                if not readable:
                    return False
                try:
                    peek = conn.recv(1, socket.MSG_PEEK)
                except BlockingIOError:
                    return False
                return peek == b""
            except Exception:
                return False

        try:
            if emit_initial_think:
                send_chunk("<think>\n")
            while True:
                if client_disconnected():
                    try:
                        self.dist_state.cancel_request(request_id)
                    except Exception:
                        pass
                    return
                try:
                    item = queue.get(timeout=10)
                except Empty:
                    if client_disconnected():
                        try:
                            self.dist_state.cancel_request(request_id)
                        except Exception:
                            pass
                        return
                    # Send a keepalive chunk to ensure streaming clients (and proxies) observe bytes
                    # even when the model is busy (e.g. long prefill) and no tokens are produced yet.
                    try:
                        send_chunk("", force=True)
                    except Exception:
                        pass
                    continue

                if item is None:
                    break

                gen_text = item.get("text", "")
                finish_reason = item.get("finish_reason")
                prompt_toks = item.get("prompt_tokens", prompt_toks)
                gen_toks = item.get("generation_tokens", gen_toks)

                if has_tool_calling and gen_text == tool_call_start:
                    in_tool_call = True
                elif in_tool_call:
                    if has_tool_calling and gen_text == tool_call_end:
                        tool_calls.append(tool_text)
                        tool_text = ""
                        in_tool_call = False
                    else:
                        tool_text += gen_text
                else:
                    content_buffer += gen_text

                if not in_tool_call and (content_buffer or tool_calls):
                    send_chunk(content_buffer, tool_calls if tool_calls else None, finish=None)
                    content_buffer = ""
                    tool_calls.clear()

            final_finish = normalize_finish_reason_for_tool_calls(finish_reason or "stop", saw_tool_calls=saw_tool_calls)

            send_chunk(
                content_buffer,
                tool_calls if tool_calls else None,
                finish=final_finish,
                force=True,
            )

            if stream_options and stream_options.get("include_usage"):
                usage_chunk = {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": prompt_toks,
                        "completion_tokens": gen_toks,
                        "total_tokens": prompt_toks + gen_toks,
                    },
                }
                self.wfile.write(f"data: {json.dumps(usage_chunk)}\n\n".encode())
                self.wfile.flush()

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected; signal generation loop to stop for this request.
            try:
                self.dist_state.cancel_request(request_id)
            except Exception:
                pass
            return

    def _blocking_chat(self, queue, request_id, model, tools, emit_initial_think: bool = False):
        content = ""
        tool_calls: List[str] = []
        tool_text = ""
        in_tool_call = False
        made_tool_call = False
        has_tool_calling = getattr(self.tokenizer, "has_tool_calling", False)
        tool_call_start = getattr(self.tokenizer, "tool_call_start", None)
        tool_call_end = getattr(self.tokenizer, "tool_call_end", None)
        tool_parser = getattr(self.tokenizer, "tool_parser", None)
        finish_reason = None
        prompt_toks = 0
        gen_toks = 0
        tool_idx = 0

        tool_parser_type = infer_tool_parser_type(self.tokenizer)
        tool_fix_ctx = ToolFixContext(
            tool_parser_type=tool_parser_type,
            tools=tools,
        )

        def parse_single_tool(tool_text: str) -> Optional[dict]:
            nonlocal tool_idx
            if tool_parser is None:
                logging.warning(
                    "Tool call emitted but tokenizer has no tool_parser (id=%s)",
                    request_id,
                )
                return None
            if not tool_text or not tool_text.strip():
                logging.warning("Skipping empty tool call text (id=%s)", request_id)
                return None
            try:
                tool_call = tool_parser(tool_text, tools)
            except Exception as e:
                preview = tool_text.strip().replace("\n", "\\n")
                if len(preview) > 400:
                    preview = preview[:400] + "…"
                logging.warning(
                    "Failed to parse tool call (id=%s): %s; preview=%r",
                    request_id,
                    e,
                    preview,
                )
                return None
            if not isinstance(tool_call, dict):
                logging.warning(
                    "Tool parser returned non-dict (id=%s): %s",
                    request_id,
                    type(tool_call).__name__,
                )
                return None

            tool_call = apply_tool_fixes(tool_call, tool_fix_ctx)
            arguments = tool_call.get("arguments", {})
            tool_call["arguments"] = json.dumps(arguments, ensure_ascii=False)
            out = make_openai_tool_call(
                name=str(tool_call.get("name") or ""),
                arguments=tool_call.get("arguments") or "{}",
                tool_call_id=str(uuid.uuid4()),
            )
            tool_idx += 1
            return out

        def parse_tools(tool_calls: List[str]) -> List[dict]:
            if not tool_calls:
                return []
            parsed = []
            for tool_text in tool_calls:
                tc = parse_single_tool(tool_text)
                if tc is not None:
                    parsed.append(tc)
            return parsed

        blocking_timeout_s = float(os.environ.get("DISTRIBUTED_BLOCKING_TIMEOUT_S", "3600"))
        blocking_poll_s = float(os.environ.get("DISTRIBUTED_BLOCKING_POLL_S", "1"))
        start_t = time.perf_counter()

        while True:
            if blocking_timeout_s > 0:
                try:
                    item = queue.get(timeout=blocking_poll_s)
                except Empty:
                    if (time.perf_counter() - start_t) >= blocking_timeout_s:
                        logging.error(
                            "Blocking chat request timed out after %.1fs (id=%s model=%s)",
                            blocking_timeout_s,
                            request_id,
                            model,
                        )
                        try:
                            self.dist_state.cancel_request(request_id)
                        except Exception:
                            pass

                        # Drain the queue in the background to avoid orphaned
                        # generation building up responses after the client has
                        # already received a timeout.
                        def _drain():
                            while True:
                                try:
                                    it = queue.get(timeout=1)
                                except Empty:
                                    continue
                                if it is None:
                                    break

                        Thread(target=_drain, daemon=True).start()
                        self._json_response(
                            504,
                            {
                                "error": f"Timed out waiting for completion after {int(blocking_timeout_s)}s",
                            },
                        )
                        return
                    continue
            else:
                item = queue.get()
            if item is None:
                break
            gen_text = item.get("text", "")
            if has_tool_calling and gen_text == tool_call_start:
                made_tool_call = True
                in_tool_call = True
            elif in_tool_call:
                if has_tool_calling and gen_text == tool_call_end:
                    tool_calls.append(tool_text)
                    tool_text = ""
                    in_tool_call = False
                else:
                    tool_text += gen_text
            else:
                content += gen_text
            finish_reason = item.get("finish_reason")
            prompt_toks = item.get("prompt_tokens", 0)
            gen_toks = item.get("generation_tokens", 0)

        tool_calls_payload = parse_tools(tool_calls)

        finish_reason = normalize_finish_reason_for_tool_calls(
            finish_reason or "stop",
            saw_tool_calls=bool(tool_calls_payload),
        )

        if emit_initial_think and not content.lstrip().startswith("<think>"):
            content = "<think>\n" + content

        response = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    **({"tool_calls": tool_calls_payload} if tool_calls_payload else {}),
                },
                "finish_reason": finish_reason or "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_toks,
                "completion_tokens": gen_toks,
                "total_tokens": prompt_toks + gen_toks,
            },
        }
        self._json_response(200, response)

    def _handle_text(self):
        body = self._parse_body()
        prompt = body.get("prompt", "")
        stream = body.get("stream", False)
        max_tokens = self._parse_max_tokens(body)
        temperature, top_p, top_k, repetition_penalty, repetition_context_size = self._parse_sampling(body)
        seed, seed_is_user = self._parse_seed(body)
        stop_token_sequences = self._parse_stop_token_sequences(body.get("stop") or [])
        model = body.get("model", self.args.model)

        prompt_tokens = self.tokenizer.encode(prompt)

        response_queue = Queue()
        request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        self.dist_state.request_queue.put({
            "request_id": request_id,
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens,
            "seed": seed,
            "seed_is_user": seed_is_user,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "repetition_context_size": repetition_context_size,
            "stop_token_sequences": stop_token_sequences,
            "response_queue": response_queue,
            "tools": None,
        })

        if stream:
            self._stream_text(response_queue, request_id, model)
        else:
            self._blocking_text(response_queue, request_id, model)

    def _stream_text(self, queue, request_id, model):
        self._stream_response()
        try:
            self.wfile.write(b": keepalive\n\n")
            self.wfile.flush()
        except Exception:
            pass

        try:
            while True:
                try:
                    item = queue.get(timeout=10)
                except Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue

                if item is None:
                    break

                chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{"index": 0, "text": item.get("text", ""), "finish_reason": item.get("finish_reason")}],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            try:
                self.dist_state.cancel_request(request_id)
            except Exception:
                pass
            return

    def _blocking_text(self, queue, request_id, model):
        full_text = ""
        finish_reason = None
        prompt_toks = 0
        gen_toks = 0

        blocking_timeout_s = float(os.environ.get("DISTRIBUTED_BLOCKING_TIMEOUT_S", "3600"))
        blocking_poll_s = float(os.environ.get("DISTRIBUTED_BLOCKING_POLL_S", "1"))
        start_t = time.perf_counter()

        while True:
            if blocking_timeout_s > 0:
                try:
                    item = queue.get(timeout=blocking_poll_s)
                except Empty:
                    if (time.perf_counter() - start_t) >= blocking_timeout_s:
                        logging.error(
                            "Blocking text request timed out after %.1fs (id=%s model=%s)",
                            blocking_timeout_s,
                            request_id,
                            model,
                        )
                        self._json_response(
                            504,
                            {
                                "error": f"Timed out waiting for completion after {int(blocking_timeout_s)}s",
                            },
                        )
                        return
                    continue
            else:
                item = queue.get()
            if item is None:
                break
            full_text += item.get("text", "")
            finish_reason = item.get("finish_reason")
            prompt_toks = item.get("prompt_tokens", 0)
            gen_toks = item.get("generation_tokens", 0)

        response = {
            "id": request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "text": full_text, "finish_reason": finish_reason or "stop"}],
            "usage": {"prompt_tokens": prompt_toks, "completion_tokens": gen_toks, "total_tokens": prompt_toks + gen_toks},
        }
        self._json_response(200, response)

    def _handle_anthropic(self):
        body = self._parse_body()
        messages = convert_anthropic_to_openai_messages(body)
        tools = convert_anthropic_tools(body.get("tools"))

        stream = body.get("stream", False)
        max_tokens = self._parse_max_tokens(body)
        temperature, top_p, top_k, repetition_penalty, repetition_context_size = self._parse_sampling(body)
        seed, seed_is_user = self._parse_seed(body)
        stop_token_sequences = self._parse_stop_token_sequences(body.get("stop_sequences") or [])
        model = body.get("model", self.args.model)

        process_message_content(messages)
        prompt = apply_chat_template_safe(
            self.tokenizer,
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )
        emit_initial_think = prompt.rstrip().endswith("<think>")
        prompt_tokens = self.tokenizer.encode(prompt)

        response_queue = Queue()
        request_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.dist_state.request_queue.put({
            "request_id": request_id,
            "prompt_tokens": prompt_tokens,
            "max_tokens": max_tokens,
            "seed": seed,
            "seed_is_user": seed_is_user,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "repetition_context_size": repetition_context_size,
            "stop_token_sequences": stop_token_sequences,
            "response_queue": response_queue,
            "tools": tools,
        })

        if stream:
            self._stream_anthropic(response_queue, request_id, model, tools, emit_initial_think)
        else:
            self._blocking_anthropic(response_queue, request_id, model, tools, emit_initial_think)

    def _stream_anthropic(self, queue, request_id, model, tools, emit_initial_think: bool = False):
        self._stream_response()

        full_text = ""
        tool_calls = []
        tool_text = ""
        in_tool_call = False
        has_tool_calling = getattr(self.tokenizer, "has_tool_calling", False)
        tool_call_start = getattr(self.tokenizer, "tool_call_start", None)
        tool_call_end = getattr(self.tokenizer, "tool_call_end", None)
        finish_reason = None
        prompt_toks = 0
        gen_toks = 0

        try:
            while True:
                try:
                    item = queue.get(timeout=60)
                except Empty:
                    try:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        raise
                    continue
                if item is None:
                    break

                gen_text = item.get("text", "")
                if has_tool_calling and gen_text == tool_call_start:
                    in_tool_call = True
                elif in_tool_call:
                    if gen_text == tool_call_end:
                        tool_calls.append(tool_text)
                        tool_text = ""
                        in_tool_call = False
                    else:
                        tool_text += gen_text
                else:
                    full_text += gen_text

                finish_reason = item.get("finish_reason")
                prompt_toks = item.get("prompt_tokens", prompt_toks)
                gen_toks = item.get("generation_tokens", gen_toks)

            self._send_anthropic_stream_events(
                full_text,
                finish_reason,
                tool_calls,
                prompt_toks,
                gen_toks,
                tools,
                request_id,
                model,
                emit_initial_think,
            )
        except (BrokenPipeError, ConnectionResetError):
            try:
                self.dist_state.cancel_request(request_id)
            except Exception:
                pass
            return

    def _send_anthropic_stream_events(
        self,
        text: str,
        finish_reason: Optional[str],
        tool_calls: Optional[List[str]],
        input_tokens: int,
        output_tokens: int,
        tools: Optional[List[dict]],
        request_id: str,
        model: str,
        emit_initial_think: bool = False,
    ):
        tool_parser = getattr(self.tokenizer, "tool_parser", None)
        tool_parser_type = infer_tool_parser_type(self.tokenizer)
        tool_fix_ctx = ToolFixContext(
            tool_parser_type=tool_parser_type,
            tools=tools,
        )
        clean_text = text
        if emit_initial_think and not clean_text.lstrip().startswith("<think>"):
            clean_text = "<think>\n" + clean_text

        parsed_tool_calls = []
        if tools and tool_parser:
            tool_call_texts = list(tool_calls) if tool_calls else [text]
            for t in tool_call_texts:
                try:
                    tc = tool_parser(t, tools)
                except Exception:
                    continue
                if not isinstance(tc, dict):
                    continue
                parsed_tool_calls.append(apply_tool_fixes(tc, tool_fix_ctx))
                if not tool_calls:
                    break

        if parsed_tool_calls:
            finish_reason = "tool_calls"

        def write_event(event: str, payload: dict) -> None:
            self.wfile.write(
                f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
            )

        write_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": request_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0},
                },
            },
        )

        write_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )

        if clean_text:
            write_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": clean_text},
                },
            )

        write_event(
            "content_block_stop",
            {
                "type": "content_block_stop",
                "index": 0,
            },
        )

        block_index = 1
        for tool_data in parsed_tool_calls:
            tool_id = f"toolu_{uuid.uuid4().hex[:24]}"

            write_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_data.get("name", ""),
                        "input": {},
                    },
                },
            )

            write_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(tool_data.get("arguments", {}), ensure_ascii=False),
                    },
                },
            )

            write_event(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": block_index,
                },
            )
            block_index += 1

        stop_reason_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}
        write_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason_map.get(finish_reason, finish_reason), "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        )

        self.wfile.write(f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n".encode())
        self.wfile.flush()

    def _blocking_anthropic(self, queue, request_id, model, tools, emit_initial_think: bool = False):
        full_text = ""
        tool_calls = []
        tool_text = ""
        in_tool_call = False
        has_tool_calling = getattr(self.tokenizer, "has_tool_calling", False)
        tool_call_start = getattr(self.tokenizer, "tool_call_start", None)
        tool_call_end = getattr(self.tokenizer, "tool_call_end", None)
        finish_reason = None
        prompt_toks = 0
        gen_toks = 0
        tool_parser_type = infer_tool_parser_type(self.tokenizer)
        tool_fix_ctx = ToolFixContext(
            tool_parser_type=tool_parser_type,
            tools=tools,
        )

        while True:
            try:
                item = queue.get(timeout=120)
            except Empty:
                break
            if item is None:
                break
            gen_text = item.get("text", "")
            if has_tool_calling and gen_text == tool_call_start:
                in_tool_call = True
            elif in_tool_call:
                if gen_text == tool_call_end:
                    tool_calls.append(tool_text)
                    tool_text = ""
                    in_tool_call = False
                else:
                    tool_text += gen_text
            else:
                full_text += gen_text
            finish_reason = item.get("finish_reason")
            prompt_toks = item.get("prompt_tokens", 0)
            gen_toks = item.get("generation_tokens", 0)

        clean_text = full_text
        if emit_initial_think and not clean_text.lstrip().startswith("<think>"):
            clean_text = "<think>\n" + clean_text
        content = []

        if clean_text:
            content.append({"type": "text", "text": clean_text})

        tool_parser = getattr(self.tokenizer, "tool_parser", None)
        parsed_tool_calls = []
        if tools and tool_parser:
            tool_call_texts = tool_calls if tool_calls else [full_text]
            for t in tool_call_texts:
                try:
                    tc = tool_parser(t, tools)
                except Exception:
                    continue
                if not isinstance(tc, dict):
                    continue
                parsed_tool_calls.append(apply_tool_fixes(tc, tool_fix_ctx))
                if not tool_calls:
                    break

        for tc in parsed_tool_calls:
            content.append({
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": tc.get("name", ""),
                    "input": tc.get("arguments", {}),
                })

        has_tool = any(b.get("type") == "tool_use" for b in content)
        stop_reason = "tool_use" if has_tool else ("end_turn" if finish_reason == "stop" else "max_tokens")

        response = {
            "id": request_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "usage": {"input_tokens": prompt_toks, "output_tokens": gen_toks},
        }
        self._json_response(200, response)




def run_http_server(dist_state, tokenizer, args):
    """Run HTTP server (rank 0 only)."""
    def factory(*a, **kw):
        return DistributedHandler(dist_state, tokenizer, args, *a, **kw)

    server_address = (args.host, args.port)
    infos = socket.getaddrinfo(
        *server_address, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )
    ThreadingHTTPServer.address_family, _, _, _, server_address = next(iter(infos))

    server = ThreadingHTTPServer(server_address, factory)
    logging.info(f"HTTP server on {args.host}:{args.port}")

    server.serve_forever()


__all__ = ["BadRequestError", "DistributedHandler", "run_http_server"]
