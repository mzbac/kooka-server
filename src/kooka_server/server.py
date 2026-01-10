import argparse
import json
import logging
import socket
import uuid
import warnings
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse

from mlx_lm.server import (
    APIHandler,
    CompletionRequest,
    GenerationArguments,
    LRUPromptCache,
    LogitsProcessorArguments,
    ModelDescription,
    ResponseGenerator,
    SamplingArguments,
    ThreadingHTTPServer,
    get_system_fingerprint,
    stopping_criteria,
)

from .api.anthropic.messages import (
    convert_anthropic_to_openai_messages,
    convert_anthropic_tools,
    process_message_content,
)
from .api.models_endpoint import json_response as models_json_response
from .api.openai.tool_calls import (
    apply_tool_fixes_to_openai_tool_calls,
    make_openai_tool_call,
    normalize_finish_reason_for_tool_calls,
    parse_json_tool_calls,
    tool_names_from_openai_tools,
)
from .logging_utils import redact_request_body
from .mlx_utils.tokenizer_compat import maybe_patch_tool_parser
from .mlx_utils.model_provider import KookaModelProvider
from .tool_fixes import ToolFixContext, apply as apply_tool_fixes, infer_tool_parser_type


def _json_error(handler: APIHandler, status: int, message: str) -> None:
    try:
        handler._set_completion_headers(status)
        handler.end_headers()
        handler.wfile.write(json.dumps({"error": message}).encode("utf-8"))
        handler.wfile.flush()
    except Exception:
        return


class KookaAPIHandler(APIHandler):
    """APIHandler wrapper with stricter request parsing."""

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        request_factories: Dict[str, Callable[[], Any]] = {
            "/v1/completions": self.handle_text_completions,
            "/v1/chat/completions": self.handle_chat_completions,
            "/chat/completions": self.handle_chat_completions,
            "/v1/messages": self.handle_anthropic_messages,
        }

        # Strip query strings (e.g. `/v1/messages?beta=true`).
        parsed_path = urlparse(self.path).path

        if parsed_path not in request_factories:
            self._set_completion_headers(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except (TypeError, ValueError):
            _json_error(self, 400, "Invalid Content-Length header")
            return

        if content_length <= 0:
            _json_error(self, 400, "Missing request body")
            return

        raw_body = self.rfile.read(content_length)
        try:
            self.body = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as e:
            logging.error("Invalid JSON body: %s", e)
            _json_error(self, 400, f"Invalid JSON in request body: {e}")
            return

        if not isinstance(self.body, dict):
            _json_error(self, 400, f"Request should be a JSON object, got {type(self.body).__name__}")
            return

        try:
            # Track tool calls across streaming chunks.
            self._saw_tool_calls = False
            self._json_tool_call_buffer = ""

            indent = "\t"
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(
                    "Incoming Request Body (redacted): %s",
                    json.dumps(redact_request_body(self.body), indent=indent, ensure_ascii=False),
                )

            self.stream = bool(self.body.get("stream", False))
            self.stream_options = self.body.get("stream_options", None)
            requested_model = self.body.get("model", "default_model")
            if isinstance(requested_model, str) and requested_model.startswith(("claude-", "anthropic")):
                self.requested_model = "default_model"
            else:
                self.requested_model = requested_model
            self.requested_draft_model = self.body.get("draft_model", "default_model")
            self.num_draft_tokens = int(
                self.body.get("num_draft_tokens", self.response_generator.cli_args.num_draft_tokens)
            )
            self.adapter = self.body.get("adapters", None)

            self.max_tokens = self.body.get("max_completion_tokens", None)
            if self.max_tokens is None:
                self.max_tokens = self.body.get("max_tokens", self.response_generator.cli_args.max_tokens)
            self.max_tokens = int(self.max_tokens)

            self.temperature = float(self.body.get("temperature", self.response_generator.cli_args.temp))
            self.top_p = float(self.body.get("top_p", self.response_generator.cli_args.top_p))
            self.top_k = int(self.body.get("top_k", self.response_generator.cli_args.top_k))
            self.min_p = float(self.body.get("min_p", self.response_generator.cli_args.min_p))
            self.repetition_penalty = float(self.body.get("repetition_penalty", 0.0))
            self.repetition_context_size = int(self.body.get("repetition_context_size", 20))
            self.xtc_probability = float(self.body.get("xtc_probability", 0.0))
            self.xtc_threshold = float(self.body.get("xtc_threshold", 0.0))
            self.logit_bias = self.body.get("logit_bias", None)
            self.logprobs = int(self.body.get("logprobs", -1))
            self.seed = self.body.get("seed", None)

            self.validate_model_parameters()

            stop_words_key = "stop_sequences" if parsed_path == "/v1/messages" else "stop"
            stop_words = self.body.get(stop_words_key) or []
            if isinstance(stop_words, str):
                stop_words = [stop_words]
            if not isinstance(stop_words, list):
                stop_words = []

            request = request_factories[parsed_path]()
            self._request_tools = getattr(request, "tools", None)
            if parsed_path == "/v1/messages":
                self.handle_anthropic_completion(request, stop_words)
            else:
                self.handle_completion(request, stop_words)
        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected mid-response (common for streaming/UIs). Avoid logging as 500.
            return
        except (AssertionError, ValueError, TypeError) as e:
            _json_error(self, 400, str(e))
            return
        except Exception:
            logging.exception("Unhandled error in request handler")
            _json_error(self, 500, "Internal server error")
            return

    def handle_models_request(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        self._set_completion_headers(200)
        self.end_headers()

        response_json = models_json_response(
            created=self.created,
            active_model=self.response_generator.cli_args.model,
            request_path=self.path,
        )
        self.wfile.write(response_json)
        self.wfile.flush()

    def handle_anthropic_messages(self) -> CompletionRequest:
        body = self.body
        messages = convert_anthropic_to_openai_messages(body)
        tools = convert_anthropic_tools(body.get("tools"))
        process_message_content(messages)

        self.request_id = f"msg_{uuid.uuid4().hex[:24]}"
        return CompletionRequest(
            "chat",
            "",
            messages,
            tools,
            None,
        )

    def _send_anthropic_events(
        self,
        *,
        request_id: str,
        model: str,
        text: str,
        finish_reason: Optional[str],
        tool_calls: list[str],
        input_tokens: int,
        output_tokens: int,
        tools: Optional[list[dict]],
    ) -> None:
        tokenizer = getattr(self.response_generator.model_provider, "tokenizer", None)
        if tokenizer is not None:
            maybe_patch_tool_parser(tokenizer)
            tool_parser = getattr(tokenizer, "tool_parser", None)
            tool_parser_type = infer_tool_parser_type(tokenizer)
        else:
            tool_parser = None
            tool_parser_type = None

        tool_fix_ctx = ToolFixContext(
            tool_parser_type=tool_parser_type,
            tools=tools,
        )

        parsed_tool_calls: list[dict] = []
        if tool_calls and callable(tool_parser):
            for t in tool_calls:
                try:
                    parsed_tool_calls.append(apply_tool_fixes(tool_parser(t, tools), tool_fix_ctx))
                except Exception:
                    logging.debug("Failed to parse tool call text", exc_info=True)

        clean_text = text
        if clean_text and parsed_tool_calls and tool_parser_type == "minimax_m2":
            stripped = clean_text.strip()
            if stripped.startswith("<invoke") and stripped.endswith("</invoke>"):
                clean_text = ""

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

        stop_reason_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "tool_call": "tool_use"}
        write_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason_map.get(finish_reason, finish_reason), "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        )

        self.wfile.write(f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n".encode("utf-8"))
        self.wfile.flush()

    def handle_anthropic_completion(self, request: CompletionRequest, stop_words: list[str]) -> None:
        args = GenerationArguments(
            model=ModelDescription(
                model=self.requested_model,
                draft=self.requested_draft_model,
                adapter=self.adapter,
            ),
            sampling=SamplingArguments(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                xtc_probability=self.xtc_probability,
                xtc_threshold=self.xtc_threshold,
            ),
            logits=LogitsProcessorArguments(
                logit_bias=self.logit_bias,
                repetition_penalty=self.repetition_penalty,
                repetition_context_size=self.repetition_context_size,
            ),
            stop_words=stop_words,
            max_tokens=self.max_tokens,
            num_draft_tokens=self.num_draft_tokens,
            logprobs=self.logprobs,
            seed=self.seed,
        )

        def keepalive_callback(processed_tokens: int, total_tokens: int) -> None:
            if not self.stream:
                return
            try:
                self.wfile.write(f": keepalive {processed_tokens}/{total_tokens}\n\n".encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return

        ctx, response = self.response_generator.generate(
            request,
            args,
            progress_callback=keepalive_callback,
        )

        in_reasoning = False
        if ctx.has_thinking:
            for i in range(len(ctx.prompt) - 1, -1, -1):
                if ctx.prompt[i] == ctx.think_end_id:
                    break
                if ctx.prompt[i] == ctx.think_start_id:
                    in_reasoning = True
                    break

        if self.stream:
            self._set_stream_headers(200)
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
        else:
            self._set_completion_headers(200)

        in_tool_call = False
        made_tool_call = False
        tool_calls: list[str] = []
        tool_text = ""
        reasoning_text = ""

        tokens: list[int] = []
        text = ""
        finish_reason: Optional[str] = "length"

        for gen in response:
            if in_reasoning:
                if gen.text == ctx.think_end:
                    in_reasoning = False
                else:
                    reasoning_text += gen.text
            elif ctx.has_tool_calling and gen.text == ctx.tool_call_start:
                made_tool_call = True
                in_tool_call = True
            elif in_tool_call:
                if gen.text == ctx.tool_call_end:
                    tool_calls.append(tool_text)
                    tool_text = ""
                    in_tool_call = False
                else:
                    tool_text += gen.text
            else:
                text += gen.text

            tokens.append(gen.token)

            stop_condition = stopping_criteria(tokens, ctx.eos_token_ids, ctx.stop_token_sequences, stop_words)
            if stop_condition.stop_met:
                finish_reason = "tool_call" if made_tool_call else "stop"
                ctx.stop()
                tokens = tokens[: len(tokens) - stop_condition.trim_length]
                text = text[: len(text) - stop_condition.trim_text_length]
                break

            if gen.finish_reason is not None:
                finish_reason = gen.finish_reason

        clean_text = text
        if reasoning_text:
            if clean_text:
                clean_text = f"<think>\n{reasoning_text}\n{ctx.think_end}\n{clean_text}"
            else:
                clean_text = f"<think>\n{reasoning_text}\n{ctx.think_end}"
        elif in_reasoning and not clean_text.lstrip().startswith("<think>"):
            clean_text = "<think>\n" + clean_text

        tools = request.tools
        if not tool_calls and tools:
            tokenizer = getattr(self.response_generator.model_provider, "tokenizer", None)
            if tokenizer is not None:
                tool_parser = getattr(tokenizer, "tool_parser", None)
                if callable(tool_parser):
                    try:
                        parsed = tool_parser(clean_text, tools)
                    except Exception:
                        parsed = None
                    if isinstance(parsed, dict) and parsed.get("name"):
                        tool_calls = [clean_text]

        if self.stream:
            self._send_anthropic_events(
                request_id=self.request_id,
                model=self.requested_model,
                text=clean_text,
                finish_reason=finish_reason,
                tool_calls=tool_calls,
                input_tokens=len(ctx.prompt),
                output_tokens=len(tokens),
                tools=tools,
            )
            return

        tokenizer = getattr(self.response_generator.model_provider, "tokenizer", None)
        if tokenizer is not None:
            maybe_patch_tool_parser(tokenizer)
            tool_parser = getattr(tokenizer, "tool_parser", None)
            tool_parser_type = infer_tool_parser_type(tokenizer)
        else:
            tool_parser = None
            tool_parser_type = None

        tool_fix_ctx = ToolFixContext(
            tool_parser_type=tool_parser_type,
            tools=tools,
        )

        content: list[dict] = []
        if clean_text:
            content.append({"type": "text", "text": clean_text})

        if tool_calls and callable(tool_parser):
            for t in tool_calls:
                try:
                    tc = apply_tool_fixes(tool_parser(t, tools), tool_fix_ctx)
                except Exception:
                    logging.debug("Failed to parse tool call text", exc_info=True)
                    continue
                content.append(
                    {
                        "type": "tool_use",
                        "id": f"toolu_{uuid.uuid4().hex[:24]}",
                        "name": tc.get("name", ""),
                        "input": tc.get("arguments", {}),
                    }
                )

        if tool_parser_type == "minimax_m2" and content:
            tool_blocks = [b for b in content if b.get("type") == "tool_use"]
            if tool_blocks and len(content) == 2 and content[0].get("type") == "text":
                txt = content[0].get("text", "")
                if isinstance(txt, str):
                    stripped = txt.strip()
                    if stripped.startswith("<invoke") and stripped.endswith("</invoke>"):
                        content = tool_blocks

        stop_reason = None
        if any(b.get("type") == "tool_use" for b in content):
            stop_reason = "tool_use"
        elif finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"

        response = {
            "id": self.request_id,
            "type": "message",
            "role": "assistant",
            "model": self.requested_model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": len(ctx.prompt), "output_tokens": len(tokens)},
        }

        response_json = json.dumps(response, ensure_ascii=False).encode("utf-8")
        self.send_header("Content-Length", str(len(response_json)))
        self.end_headers()
        self.wfile.write(response_json)
        self.wfile.flush()

    def generate_response(  # noqa: PLR0913
        self,
        text: str,
        finish_reason: Any,
        prompt_token_count: Optional[int] = None,
        completion_token_count: Optional[int] = None,
        token_logprobs: Optional[list] = None,
        top_tokens: Optional[list] = None,
        tokens: Optional[list] = None,
        tool_calls: Optional[list] = None,
        reasoning_text: Optional[str] = None,
    ) -> dict:
        tool_calls_for_super = tool_calls
        if tool_calls and any(isinstance(tc, str) for tc in tool_calls):
            tokenizer = getattr(self.response_generator.model_provider, "tokenizer", None)
            if tokenizer is not None:
                maybe_patch_tool_parser(tokenizer)
                tool_parser = getattr(tokenizer, "tool_parser", None)
            else:
                tool_parser = None

            parsed: list[dict] = []
            tool_idx = 0
            for tc in tool_calls:
                if isinstance(tc, dict):
                    parsed.append(tc)
                    continue
                if not isinstance(tc, str):
                    continue
                tc_s = tc.strip()
                if not tc_s:
                    continue

                call_obj: Any = None
                try:
                    call_obj = json.loads(tc_s)
                except (json.JSONDecodeError, ValueError):
                    call_obj = None

                if call_obj is None and callable(tool_parser):
                    try:
                        call_obj = tool_parser(tc_s, getattr(self, "_request_tools", None))
                    except Exception:
                        logging.debug("Failed to parse tool call text", exc_info=True)
                        call_obj = None

                if not isinstance(call_obj, dict):
                    continue

                arguments = call_obj.get("arguments")
                if not isinstance(arguments, str):
                    call_obj = dict(call_obj)
                    call_obj["arguments"] = json.dumps(arguments, ensure_ascii=False)

                parsed.append(
                    make_openai_tool_call(
                        name=str(call_obj.get("name") or ""),
                        arguments=call_obj.get("arguments") or "{}",
                        tool_call_id=str(uuid.uuid4()),
                        index=tool_idx if self.stream else None,
                    )
                )
                if self.stream:
                    tool_idx += 1

            tool_calls_for_super = parsed if parsed else tool_calls

        try:
            response = super().generate_response(
                text,
                finish_reason,
                prompt_token_count=prompt_token_count,
                completion_token_count=completion_token_count,
                token_logprobs=token_logprobs,
                top_tokens=top_tokens,
                tokens=tokens,
                tool_calls=tool_calls_for_super,
                reasoning_text=reasoning_text,
            )
        except TypeError:
            response = super().generate_response(
                text,
                finish_reason,
                prompt_token_count=prompt_token_count,
                completion_token_count=completion_token_count,
                token_logprobs=token_logprobs,
                top_tokens=top_tokens,
                tokens=tokens,
                tool_calls=tool_calls_for_super,
            )

        choice = None
        if isinstance(response, dict):
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                choice = choices[0]
        if isinstance(choice, dict) and response.get("object", "").startswith("chat.completion"):
            key_name = "delta" if self.stream else "message"
            msg = choice.get(key_name)
            if isinstance(msg, dict):
                tool_calls_obj = msg.get("tool_calls")
                if tool_calls_obj is None:
                    tool_calls_obj = []
                    msg["tool_calls"] = tool_calls_obj

                tools = getattr(self, "_request_tools", None)
                if not tool_calls_obj and isinstance(tools, list):
                    allowed_names = tool_names_from_openai_tools(tools)
                    content = msg.get("content")
                    if self.stream:
                        if isinstance(content, str):
                            self._json_tool_call_buffer = getattr(self, "_json_tool_call_buffer", "") + content
                        if choice.get("finish_reason") is not None:
                            parsed = parse_json_tool_calls(getattr(self, "_json_tool_call_buffer", ""), allowed_names)
                            if parsed:
                                msg["content"] = ""
                                msg["tool_calls"] = parsed
                                tool_calls_obj = parsed
                                self._json_tool_call_buffer = ""
                                if choice.get("finish_reason") in {"stop", "tool_call"}:
                                    choice["finish_reason"] = "tool_calls"
                    else:
                        if isinstance(content, str) and content:
                            parsed = parse_json_tool_calls(content, allowed_names)
                            if parsed:
                                msg["content"] = ""
                                msg["tool_calls"] = parsed
                                tool_calls_obj = parsed
                                if choice.get("finish_reason") in {"stop", "tool_call"}:
                                    choice["finish_reason"] = "tool_calls"

                if isinstance(tool_calls_obj, list) and tool_calls_obj:
                    self._saw_tool_calls = True

                choice["finish_reason"] = normalize_finish_reason_for_tool_calls(
                    choice.get("finish_reason"),
                    saw_tool_calls=bool(getattr(self, "_saw_tool_calls", False)),
                )
                if not self.stream and choice.get("finish_reason") == "tool_calls" and not tool_calls_obj:
                    choice["finish_reason"] = "stop"

                if tool_calls_obj:
                    tokenizer = getattr(self.response_generator.model_provider, "tokenizer", None)
                    msg["tool_calls"] = apply_tool_fixes_to_openai_tool_calls(
                        tool_calls_obj,
                        tool_parser_type=infer_tool_parser_type(tokenizer),
                        tools=getattr(self, "_request_tools", None),
                    )

        return response


def serve(args: argparse.Namespace) -> None:
    """Run a single-machine server."""
    model_provider = KookaModelProvider(args)
    response_generator = ResponseGenerator(model_provider, LRUPromptCache())
    server_address = (args.host, args.port)

    infos = socket.getaddrinfo(*server_address, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)
    ThreadingHTTPServer.address_family, _, _, _, server_address = next(iter(infos))

    httpd = ThreadingHTTPServer(
        server_address,
        lambda *a, **kw: KookaAPIHandler(
            response_generator,
            system_fingerprint=get_system_fingerprint(),
            *a,
            **kw,
        ),
    )
    warnings.warn("kooka-server: early development; contract enforced by pytest contract tests (tests/).")
    logging.info("Starting kooka-server at %s:%d", args.host, args.port)
    httpd.serve_forever()
