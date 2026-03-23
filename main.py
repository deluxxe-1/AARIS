import json
import os
import re
import difflib
from typing import Any
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from ollama import chat

from tools import (
    append_file,
    create_file,
    create_folder,
    copy_path,
    detect_project,
    build_text_index,
    rag_query,
    project_workflow_suggest,
    delete_path,
    describe_path,
    count_dir_children_matches,
    service_status,
    service_restart,
    service_health_report,
    service_restart_with_deps,
    service_wait_active,
    apply_unified_patch,
    install_packages,
    disk_usage,
    edit_file,
    fuzzy_search_paths,
    glob_find,
    estimate_dir,
    apply_template,
    exists_path,
    list_directory,
    list_processes,
    insert_after,
    read_file,
    move_path,
    resolve_path,
    rollback,
    rollback_tokens,
    run_command,
    run_command_checked,
    run_command_retry,
    search_replace_in_file,
    tail_file,
    ast_list_functions,
    ast_read_function,
    docker_ps,
    docker_logs,
    docker_exec,
    db_query_sqlite,
    delegate_task,
    schedule_agent_task,
    policy_show,
    policy_set,
    policy_reset,
)

console = Console()

MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
MAX_TOOL_ROUNDS = int(os.environ.get("AARIS_MAX_TOOL_ROUNDS", "12"))
MAX_CONTEXT_MESSAGES = int(os.environ.get("AARIS_MAX_CONTEXT_MESSAGES", "20"))
PLAN_MODE = os.environ.get("AARIS_PLAN_MODE", "off")  # off | auto | confirm
DRY_RUN = os.environ.get("AARIS_DRY_RUN", "false").strip().lower() in ("1", "true", "yes", "si", "sí", "on")
PREVIEW_MUTATIONS = os.environ.get("AARIS_PREVIEW_MUTATIONS", "true").strip().lower() in ("1", "true", "yes", "si", "sí", "on")
PREVIEW_CONFIRM_ALWAYS = os.environ.get("AARIS_PREVIEW_CONFIRM_ALWAYS", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "si",
    "sí",
    "on",
)
DIFF_MAX_LINES = int(os.environ.get("AARIS_DIFF_MAX_LINES", "300"))

DEFAULT_MEMORY_PATH = os.environ.get(
    "AARIS_MEMORY_PATH",
    os.path.join(os.path.expanduser("~"), ".aaris", "memory.json"),
)

DEFAULT_LOG_PATH = os.environ.get(
    "AARIS_LOG_PATH",
    os.path.join(os.path.expanduser("~"), ".aaris", "agent_log.jsonl"),
)

DEFAULT_APP_DIR = os.environ.get("AARIS_APP_DIR", os.path.join(os.getcwd(), ".aaris"))
UNDO_REDO_PATH = os.environ.get("AARIS_UNDO_REDO_PATH", os.path.join(DEFAULT_APP_DIR, "undo_redo.json"))


def _chat_options() -> dict[str, Any]:
    opts: dict[str, Any] = {}
    if ctx := os.environ.get("OLLAMA_NUM_CTX"):
        try:
            opts["num_ctx"] = int(ctx)
        except ValueError:
            pass
    if t := os.environ.get("OLLAMA_TEMPERATURE"):
        try:
            opts["temperature"] = float(t)
        except ValueError:
            pass
    return opts


def _load_undo_redo_state() -> dict[str, Any]:
    try:
        p = Path(UNDO_REDO_PATH).expanduser()
        if not p.is_file():
            return {"undo": [], "redo": []}
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"undo": [], "redo": []}
        data.setdefault("undo", [])
        data.setdefault("redo", [])
        return data
    except Exception:
        return {"undo": [], "redo": []}


def _save_undo_redo_state(state: dict[str, Any]) -> None:
    try:
        p = Path(UNDO_REDO_PATH).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        pass


SYSTEM_PROMPT = """Eres AARIS, un asistente de sistema para Linux. Objetivo: resolver tareas del usuario de forma fiable usando herramientas cuando aporten datos o acciones concretas.

## Reglas
- Responde en español, claro y directo.
- Para crear/editar/leer archivos, listar carpetas, buscar rutas o ejecutar comandos del sistema, usa SIEMPRE las herramientas disponibles en lugar de inventar resultados.
- Antes de editar un archivo grande, lee su contenido con read_file o usa search_replace_in_file para cambios localizados.
- Si el usuario menciona rutas “humanas” como `Documents`, `Descargas`, `Escritorio` o similares, usa `resolve_path` para mapearlas a la carpeta real dentro del `$HOME`.
- Para borrar usa `delete_path` (mueve a Trash si está activo). Si es una carpeta con `recursive=true`, confirma (`confirm=true`) siempre.
- Para borrados recursivos en carpetas muy grandes, pasa `glob_filter` para borrar solo partes (o usa una subruta más específica).
- Para instalar paquetes del sistema, usa `install_packages` y respeta `confirm=true` cuando aplique.
- Para aplicar parches unificados, usa `apply_unified_patch` (con `confirm=true` en modo seguro).
- Para búsquedas RAG locales, usa `build_text_index` y luego `rag_query` (cacheado en disco).
- Para cambios de texto pequeños, prefiere `append_file` o `insert_after` antes de reescribir archivos completos.
- Para mover/copia de recursos, usa `move_path` y `copy_path`.
- Antes de acciones con alto impacto, usa `describe_path`/`estimate_dir` para previsualizar.
- run_command ejecuta shell real: no uses comandos destructivos salvo que el usuario lo pida explícitamente; respeta su intención. Si hace falta, pasa `allow_dangerous=true`.
- Si una herramienta falla, interpreta el mensaje de error y reintenta con argumentos corregidos o explica qué falta.
- Si no necesitas herramientas, responde en texto normal.

## Contexto
Estás en una sesión interactiva; el directorio de trabajo del proceso es el cwd del usuario al lanzar el programa. Usa rutas absolutas cuando el usuario las dé, o relativas al cwd actual."""


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_memory(memory_path: str) -> dict[str, Any]:
    try:
        p = Path(memory_path).expanduser()
        if not p.is_file():
            return {
                "memory_summary": "",
                "stable_facts": [],
                "preferences": {},
                "last_updated": "",
                "last_turns": [],
            }
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("memory.json no es un objeto JSON")
        data.setdefault("memory_summary", "")
        data.setdefault("stable_facts", [])
        data.setdefault("preferences", {})
        data.setdefault("last_updated", "")
        data.setdefault("last_turns", [])
        return data
    except Exception:
        # Si la memoria está corrupta, reiniciamos sin romper el agente.
        return {
            "memory_summary": "",
            "stable_facts": [],
            "preferences": {},
            "last_updated": "",
            "last_turns": [],
        }


def _save_memory(memory_path: str, memory: dict[str, Any]) -> None:
    p = Path(memory_path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)


def _build_prefix_messages(memory: dict[str, Any]) -> list[dict[str, Any]]:
    prefix: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if memory.get("memory_summary"):
        prefix.append(
            {
                "role": "system",
                "content": "Memoria persistente del usuario (resumen estable):\n"
                + str(memory.get("memory_summary")),
            }
        )
    if memory.get("stable_facts"):
        prefix.append(
            {
                "role": "system",
                "content": "Hechos estables del usuario:\n"
                + "\n".join(str(x) for x in (memory.get("stable_facts") or [])[:50]),
            }
        )
    if memory.get("preferences"):
        prefix.append(
            {
                "role": "system",
                "content": "Preferencias detectadas del usuario (para decidir rutas y estilo):\n"
                + json.dumps(memory.get("preferences") or {}, ensure_ascii=False),
            }
        )
    return prefix


def _prune_messages(messages: list[dict[str, Any]], keep_last: int) -> list[dict[str, Any]]:
    if len(messages) <= keep_last:
        return messages
    prefix = messages[:2] if len(messages) >= 2 and messages[0].get("role") == "system" else messages[:1]
    tail = messages[-(keep_last - len(prefix)) :]
    return prefix + tail


def _extract_recent_for_memory(messages: list[dict[str, Any]], max_items: int = 10) -> list[dict[str, Any]]:
    # Nos quedamos solo con user/assistant (ignoramos tool messages para evitar ruido y datos técnicos).
    reduced: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") in ("user", "assistant", "system"):
            reduced.append({"role": m.get("role"), "content": m.get("content", "")})
    # Queremos las últimas interacciones reales, así que cortamos por el final.
    reduced = [m for m in reduced if m["role"] in ("user", "assistant")]
    return reduced[-max_items:]


def _heuristic_requires_tools(user_input: str) -> bool:
    s = user_input.lower()
    keywords = [
        "crear",
        "editar",
        "actualizar",
        "borrar",
        "eliminar",
        "borra",
        "carpeta",
        "directorio",
        "archivo",
        "comando",
        "ejecuta",
        "instalar",
        "rm ",
        "cp ",
        "mv ",
        "sudo ",
    ]
    return any(k in s for k in keywords)


def _plan_turn(user_input: str, messages: list[dict[str, Any]], opts: dict[str, Any]) -> dict[str, Any]:
    planning_user = {
        "role": "user",
        "content": (
            "Necesito un PLAN ANTES de ejecutar acciones con herramientas.\n"
            "Devuelve SOLO JSON válido con estas claves:\n"
            "- requires_tools (boolean)\n"
            "- summary (string)\n"
            "- steps (lista de strings, máximo 8)\n"
            "- safety_notes (lista de strings, máximo 5)\n\n"
            f"Tarea del usuario: {user_input}"
        ),
    }

    memless = messages[:]
    planning_messages = memless + [planning_user]
    plan_opts = dict(opts)
    plan_opts["temperature"] = 0.1

    try:
        response = chat(
            model=MODEL,
            messages=planning_messages,
            options=plan_opts or None,
        )
        content = response["message"].get("content") or ""
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            if "requires_tools" not in parsed:
                parsed["requires_tools"] = _heuristic_requires_tools(user_input)
            return parsed
    except Exception:
        pass

    return {
        "requires_tools": _heuristic_requires_tools(user_input),
        "summary": "Plan aproximado (sin respuesta JSON válida).",
        "steps": [],
        "safety_notes": [],
    }


def _update_memory(
    messages: list[dict[str, Any]],
    memory: dict[str, Any],
    opts: dict[str, Any],
) -> dict[str, Any]:
    # Llamada ligera: genera un JSON pequeño con resumen estable y preferencias.
    recent = _extract_recent_for_memory(messages, max_items=10)
    if not recent:
        return memory

    mem_prompt = {
        "role": "system",
        "content": (
            "Actualiza la memoria persistente. Devuelve SOLO JSON válido con las claves: "
            "`memory_summary` (string), `stable_facts` (lista de strings), `preferences` (objeto). "
            "No incluyas markdown ni explicaciones fuera del JSON. "
            "Regla: solo agrega información que sea estable (preferencias) y un resumen corto de lo ocurrido."
        ),
    }

    user_payload = {
        "current_memory_summary": memory.get("memory_summary", ""),
        "recent_turns": recent,
    }

    mem_options = dict(opts)
    # La memoria debe ser determinista.
    mem_options["temperature"] = 0.1

    response = chat(
        model=MODEL,
        messages=[mem_prompt, {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
        options=mem_options or None,
    )
    content = response["message"].get("content") or ""

    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("JSON no es objeto")
        memory["memory_summary"] = str(parsed.get("memory_summary") or "").strip()
        memory["stable_facts"] = list(parsed.get("stable_facts") or [])
        new_prefs = parsed.get("preferences") or {}
        if isinstance(new_prefs, dict):
            memory["preferences"] = {**(memory.get("preferences") or {}), **new_prefs}
        else:
            memory["preferences"] = memory.get("preferences") or {}
        memory["last_updated"] = _now_iso()
        return memory
    except Exception:
        return memory


def _normalize_tool_arguments(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _run_tool_loop(
    messages: list,
    available_tools: list,
    tool_map: dict,
    options: dict[str, Any],
) -> str:
    rounds = 0
    reply_content = ""
    hit_round_limit = False
    rollback_tokens_accum: list[str] = []

    while rounds < MAX_TOOL_ROUNDS:
        rounds += 1
        with console.status("[bold cyan]Pensando…[/bold cyan]", spinner="dots"):
            response = chat(
                model=MODEL,
                messages=messages,
                tools=available_tools,
                options=options or None,
            )
        response_message = response["message"]
        messages.append(response_message)

        tool_calls = response_message.get("tool_calls") or []
        if not tool_calls:
            reply_content = response_message.get("content") or ""
            break

        for tool_call in tool_calls:
            fn = tool_call.get("function") or {}
            function_name = fn.get("name") or ""
            arguments = _normalize_tool_arguments(fn.get("arguments"))
            args_preview = json.dumps(arguments, ensure_ascii=False)
            if len(args_preview) > 500:
                args_preview = args_preview[:500] + "…"

            console.print(f"[dim italic]⚙ {function_name}({args_preview})[/dim italic]")

            mutation_tools = {
                "create_file",
                "edit_file",
                "search_replace_in_file",
                "create_folder",
                "append_file",
                "insert_after",
                "delete_path",
                "run_command",
                "copy_path",
                "move_path",
            }

            tool_risk = {
                "run_command": "high",
                "delete_path": "high",
                "service_restart": "high",
                "service_restart_with_deps": "high",
                "service_health_report": "low",
                "service_wait_active": "medium",
                "move_path": "medium",
                "copy_path": "medium",
                "edit_file": "medium",
                "search_replace_in_file": "medium",
                "apply_unified_patch": "high",
                "install_packages": "high",
                "append_file": "medium",
                "insert_after": "medium",
                "create_file": "medium",
                "create_folder": "low",
            }
            need_human_confirm = (
                PREVIEW_CONFIRM_ALWAYS
                or PLAN_MODE == "confirm"
                or tool_risk.get(function_name) == "high"
            )

            # Detección adicional de rutas/acciones especialmente sensibles.
            try:
                sensitive_prefixes = ["/etc/", "/usr/", "/var/"]
                home = os.path.expanduser("~")
                sensitive_substrings = ["/.ssh", "/.gnupg", "/.kube", "/.local/share", "/.config"]
                path_guess = None
                if function_name == "delete_path":
                    path_guess = arguments.get("path")
                elif function_name in ("edit_file", "search_replace_in_file", "read_file", "append_file", "insert_after", "apply_unified_patch"):
                    path_guess = arguments.get("path") or arguments.get("destination") or arguments.get("workdir")
                elif function_name in ("copy_path", "move_path"):
                    path_guess = arguments.get("to_path") or arguments.get("from_path")
                if path_guess:
                    resolved = resolve_path(str(path_guess), must_exist=True)
                    if not str(resolved).startswith("Error:"):
                        r = str(resolved)
                        if any(r.startswith(p) for p in sensitive_prefixes) or any(s in r for s in sensitive_substrings):
                            need_human_confirm = True
            except Exception:
                pass

            # Confirmación por tipo de acción.
            if function_name in ("copy_path", "move_path") and bool(arguments.get("overwrite")):
                need_human_confirm = True
            if function_name in ("apply_unified_patch", "install_packages", "service_restart"):
                if not bool(arguments.get("confirm")) and not bool(arguments.get("allow_dangerous")):
                    need_human_confirm = True
            result = ""

            if PREVIEW_MUTATIONS and function_name in mutation_tools:
                preview = ""
                try:
                    if function_name == "delete_path":
                        p = arguments.get("path") or ""
                        rec = bool(arguments.get("recursive"))
                        preview = f"Destino: {describe_path(p)}"
                        if rec:
                            preview += "\nAlcance (estimado): " + estimate_dir(p, max_entries=1000)
                            if arguments.get("glob_filter"):
                                preview += f"\nFiltro: glob_filter={arguments.get('glob_filter')!r}"
                                try:
                                    m = count_dir_children_matches(
                                        p,
                                        arguments.get("glob_filter") or "",
                                        show_hidden=False,
                                    )
                                    preview += f"\nCoincidencias inmediatas: {m}"
                                except Exception:
                                    pass
                    elif function_name in ("create_file", "edit_file", "read_file"):
                        preview = "Destino: " + describe_path(arguments.get("path") or "")
                    elif function_name == "create_folder":
                        preview = "Carpeta: " + describe_path(arguments.get("path") or "")
                    elif function_name in ("append_file", "insert_after"):
                        preview = "Destino: " + describe_path(arguments.get("path") or "")
                    elif function_name in ("copy_path", "move_path"):
                        preview = (
                            f"Origen: {describe_path(arguments.get('from_path') or '')}\n"
                            f"Destino: {describe_path(arguments.get('to_path') or '')}"
                        )
                    elif function_name == "run_command":
                        preview = f"Comando: {arguments.get('command') or ''}"
                    elif function_name == "search_replace_in_file":
                        preview = "Archivo: " + describe_path(arguments.get("path") or "")
                except Exception:
                    preview = "(sin previsualización)"

                if preview:
                    console.print(f"[dim]Preflight:[/dim]\n{preview}")

                # Preview de diff para ediciones de texto.
                try:
                    if function_name == "edit_file":
                        p = arguments.get("path") or ""
                        old = read_file(p, max_chars=80000)
                        if isinstance(old, str) and not old.startswith("Error:") and old.strip():
                            new_content = arguments.get("new_content") or ""
                            # Previsualización semántica para JSON (cambios de keys).
                            if str(p).lower().endswith(".json"):
                                try:
                                    old_obj = json.loads(old)
                                    new_obj = json.loads(new_content)
                                    if isinstance(old_obj, dict) and isinstance(new_obj, dict):
                                        old_keys = set(old_obj.keys())
                                        new_keys = set(new_obj.keys())
                                        added = sorted(list(new_keys - old_keys))[:20]
                                        removed = sorted(list(old_keys - new_keys))[:20]
                                        changed = []
                                        for k in list(old_keys & new_keys)[:50]:
                                            if old_obj.get(k) != new_obj.get(k):
                                                changed.append(k)
                                        if added or removed or changed:
                                            console.print(
                                                "[dim]JSON keys diff:[/dim] "
                                                f"added={added} removed={removed} changed_head={changed[:20]}"
                                            )
                                except Exception:
                                    pass
                            diff_lines_iter = difflib.unified_diff(
                                old.splitlines(True),
                                new_content.splitlines(True),
                                fromfile="before",
                                tofile="after",
                                n=3,
                            )
                            diff_lines_list = list(diff_lines_iter)
                            diff_text = "".join(diff_lines_list[:DIFF_MAX_LINES])
                            if diff_text.strip():
                                console.print(f"[dim]Diff (preview):[/dim]\n{diff_text}")
                            changed_lines = [
                                l
                                for l in diff_lines_list
                                if (l.startswith("+") or l.startswith("-"))
                                and not l.startswith("+++")
                                and not l.startswith("---")
                            ]
                            if len(changed_lines) > 60 or len(diff_lines_list) > 200:
                                if need_human_confirm and not DRY_RUN and not result:
                                    ans = Prompt.ask(
                                        "Este `edit_file` parece un cambio grande. ¿Confirmas? (s/N)",
                                        default="N",
                                    ).strip().lower()
                                    if ans not in ("s", "si", "sí", "y", "yes"):
                                        result = "Acción cancelada por el usuario."
                            elif len(changed_lines) <= 10 and not DRY_RUN:
                                console.print(
                                    "[dim]Sugerencia: para cambios pequeños, intenta `search_replace_in_file`/`insert_after` en vez de `edit_file` completo.[/dim]"
                                )

                    elif function_name == "search_replace_in_file":
                        p = arguments.get("path") or ""
                        old = read_file(p, max_chars=80000)
                        if isinstance(old, str) and not old.startswith("Error:"):
                            old_text = arguments.get("old_text") or ""
                            new_text = arguments.get("new_text") or ""
                            replace_all = bool(arguments.get("replace_all"))
                            if old_text in old:
                                predicted = old.replace(old_text, new_text) if replace_all else old.replace(old_text, new_text, 1)
                                diff_lines = difflib.unified_diff(
                                    old.splitlines(True),
                                    predicted.splitlines(True),
                                    fromfile="before",
                                    tofile="after",
                                    n=3,
                                )
                                diff_text = "".join(list(diff_lines)[:DIFF_MAX_LINES])
                                if diff_text.strip():
                                    console.print(f"[dim]Diff (preview):[/dim]\n{diff_text}")
                except Exception:
                    pass

            # Confirmación inteligente para borrados recursivos.
            if function_name == "delete_path" and bool(arguments.get("recursive")) and not arguments.get("confirm"):
                if need_human_confirm and not DRY_RUN:
                    p = arguments.get("path") or ""
                    console.print(f"[bold yellow]Confirmación requerida:[/bold yellow] borrado recursivo de {p}")
                    console.print("Alcance (estimado): " + estimate_dir(p, max_entries=1000))
                    ans = Prompt.ask("¿Confirmas? (s/N)", default="N").strip().lower()
                    if ans in ("s", "si", "sí", "y", "yes"):
                        arguments["confirm"] = True
                    else:
                        result = "Acción cancelada por el usuario."

                if DRY_RUN:
                    result = "DRY_RUN: cancelado por preflight (confirm=false)."

            # Confirmación inteligente para comandos peligrosos.
            if function_name == "run_command" and bool(arguments.get("allow_dangerous")) and not result:
                if need_human_confirm and not DRY_RUN:
                    cmd = arguments.get("command") or ""
                    ans = Prompt.ask(f"Ejecutar comando peligroso?\n{cmd}\n(s/N)", default="N").strip().lower()
                    if ans not in ("s", "si", "sí", "y", "yes"):
                        result = "Acción cancelada por el usuario."

            if not result and DRY_RUN and function_name in mutation_tools:
                result = "DRY_RUN: acción no ejecutada (previsualización mostrada)."

            if not result:
                if function_name in tool_map:
                    func = tool_map[function_name]
                    try:
                        result = func(**arguments)
                    except TypeError as e:
                        result = f"Error de argumentos para {function_name}: {e}"
                    except Exception as e:
                        result = f"Error: {e}"
                else:
                    result = f"Herramienta desconocida: {function_name}"

            # Resolución interactiva si `resolve_path` devuelve candidatos ambiguos.
            if (
                function_name == "resolve_path"
                and isinstance(result, str)
                and "CANDIDATES_JSON=" in result
                and not DRY_RUN
            ):
                try:
                    m = re.search(r"CANDIDATES_JSON=([\s\S]*?)\.\s*Repite", result)
                    if not m:
                        m = re.search(r"CANDIDATES_JSON=([\s\S]*)", result)
                    if m:
                        candidates = json.loads(m.group(1).strip())
                        if isinstance(candidates, list) and candidates:
                            console.print("[bold yellow]Resolución de ruta ambigua:[/bold yellow]")
                            for i, c in enumerate(candidates[:10]):
                                console.print(
                                    f"{i+1}. {c.get('name')} (score={c.get('score')}) -> {c.get('path')}"
                                )
                            auto_pref = os.environ.get("AARIS_AUTO_RESOLVE_AMBIGUOUS", "").strip().lower()
                            chosen_path = None
                            if auto_pref in ("", "none", "off"):
                                ans = Prompt.ask("Elige número", default="1").strip()
                                idx = int(ans) - 1
                                if 0 <= idx < len(candidates):
                                    chosen_path = candidates[idx].get("path")
                            elif auto_pref in ("first", "0"):
                                chosen_path = candidates[0].get("path")
                            elif auto_pref in ("best", "best_score", "mejor", "mejor_score"):
                                chosen_path = candidates[0].get("path")
                            elif auto_pref in ("mtime_recent", "mtime_newest", "reciente", "nuevo"):
                                best_ts = -1.0
                                for c in candidates:
                                    cp = c.get("path")
                                    if not cp:
                                        continue
                                    try:
                                        ts = os.path.getmtime(str(cp))
                                    except Exception:
                                        ts = -1.0
                                    if ts > best_ts:
                                        best_ts = ts
                                        chosen_path = cp
                            else:
                                ans = Prompt.ask("Elige número", default="1").strip()
                                idx = int(ans) - 1
                                if 0 <= idx < len(candidates):
                                    chosen_path = candidates[idx].get("path")

                            if chosen_path:
                                result = str(chosen_path)
                except Exception:
                    pass

            # Extraemos tokens de rollback si la tool devolvió un resultado exitoso.
            if result and not str(result).startswith("Error"):
                m = re.search(r"ROLLBACK_TOKEN=([a-fA-F0-9]+)", str(result))
                if m:
                    rollback_tokens_accum.append(m.group(1))
                m2 = re.search(r"ROLLBACK_TOKENS=([a-zA-Z0-9,-]+)", str(result))
                if m2:
                    toks = [x.strip() for x in m2.group(1).split(",") if x.strip()]
                    rollback_tokens_accum.extend(toks)

            # Si ocurre un error y tenemos tokens, hacemos rollback automático.
            if result and str(result).startswith("Error") and rollback_tokens_accum and not DRY_RUN:
                try:
                    rb_func = tool_map.get("rollback_tokens")
                    if rb_func:
                        tokens_str = ",".join(reversed(rollback_tokens_accum))
                        rb_res = rb_func(tokens=tokens_str, overwrite=True)
                        messages.append({"role": "tool", "name": "rollback_tokens", "content": str(rb_res)})
                    reply_content = (
                        "Ocurrió un error durante la ejecución de herramientas. "
                        "Se intentó un rollback automático del último estado."
                    )
                    return reply_content
                except Exception:
                    reply_content = (
                        "Ocurrió un error durante la ejecución de herramientas (rollback automático falló)."
                    )
                    return reply_content

            rpreview = str(result)
            if len(rpreview) > 1200:
                rpreview = rpreview[:1200] + "…"
            console.print(f"[dim green]→ {rpreview}[/dim green]")

            tool_content = str(result)
            if len(tool_content) > 16000:
                tool_content = tool_content[:16000] + "\n[...Truncado por seguridad de memoria...]"

            tool_msg: dict[str, Any] = {
                "role": "tool",
                "content": tool_content,
            }
            if function_name:
                tool_msg["name"] = function_name
            messages.append(tool_msg)

            if (
                str(result).startswith("Error")
                and os.environ.get("AARIS_TOOL_ERROR_HINT", "true").strip().lower()
                in ("1", "true", "yes", "si", "sí", "on")
            ):
                messages.append(
                    {
                        "role": "system",
                        "content": f"Corrige los argumentos de la tool `{function_name}` para resolver el error: {str(result)[:800]}",
                    }
                )

        if rounds >= MAX_TOOL_ROUNDS:
            hit_round_limit = True
            break

    if hit_round_limit and not reply_content.strip():
        messages.append(
            {
                "role": "user",
                "content": "Resume en español qué hiciste y qué falta; no llames más herramientas en esta respuesta.",
            }
        )
        with console.status("[bold cyan]Síntesis…[/bold cyan]", spinner="dots"):
            final = chat(
                model=MODEL,
                messages=messages,
                options=options or None,
            )
        final_msg = final["message"]
        messages.append(final_msg)
        reply_content = final_msg.get("content") or ""

    if hit_round_limit and not reply_content.strip():
        reply_content = (
            "Se alcanzó el límite de rondas de herramientas (AARIS_MAX_TOOL_ROUNDS). "
            "Repite la petición o aumenta el límite."
        )

    return reply_content


def main():
    opts = _chat_options()
    memory_path = os.path.abspath(DEFAULT_MEMORY_PATH)
    log_path = os.path.abspath(DEFAULT_LOG_PATH)
    memory = _load_memory(memory_path)
    prefix_messages = _build_prefix_messages(memory)
    messages: list[dict[str, Any]] = prefix_messages[:]

    # Si el usuario configuró un workspace en memoria, intentamos chdir.
    try:
        prefs = memory.get("preferences") or {}
        ws_root = prefs.get("workspace_root") if isinstance(prefs, dict) else None
        if ws_root:
            resolved_ws = resolve_path(str(ws_root), must_exist=True)
            if not str(resolved_ws).startswith("Error:"):
                os.chdir(str(resolved_ws))
    except Exception:
        pass

    console.print(
        Panel.fit(
            f"[bold blue]AARIS[/bold blue] — asistente local (modelo [cyan]{MODEL}[/cyan])\n"
            "Escribe salir / exit / quit para terminar.\n"
            "Comandos: `ver memoria`, `reset memoria`, `workspace show`, `set workspace <ruta>`.\n"
            f"[dim]Opciones: OLLAMA_MODEL, OLLAMA_NUM_CTX, OLLAMA_TEMPERATURE, AARIS_MAX_TOOL_ROUNDS, AARIS_MAX_CONTEXT_MESSAGES[/dim]",
            border_style="blue",
        )
    )

    available_tools = [
        create_file,
        append_file,
        apply_template,
        apply_unified_patch,
        read_file,
        edit_file,
        search_replace_in_file,
        create_folder,
        insert_after,
        copy_path,
        detect_project,
        install_packages,
        move_path,
        resolve_path,
        delete_path,
        exists_path,
        describe_path,
        estimate_dir,
        count_dir_children_matches,
        disk_usage,
        service_status,
        service_restart,
        service_wait_active,
        service_health_report,
        service_restart_with_deps,
        list_processes,
        tail_file,
        fuzzy_search_paths,
        build_text_index,
        rag_query,
        project_workflow_suggest,
        list_directory,
        glob_find,
        run_command,
        run_command_checked,
        run_command_retry,
        policy_show,
        policy_set,
        policy_reset,
        rollback,
        rollback_tokens,
        ast_list_functions,
        ast_read_function,
        docker_ps,
        docker_logs,
        docker_exec,
        db_query_sqlite,
        delegate_task,
        schedule_agent_task,
    ]

    tool_map = {f.__name__: f for f in available_tools}

    import sys
    if "--server" in sys.argv:
        from http.server import BaseHTTPRequestHandler, HTTPServer
        class AarisAPI(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == '/api/chat':
                    length = int(self.headers.get('Content-Length', '0'))
                    post_data = self.rfile.read(length)
                    data = json.loads(post_data.decode('utf-8'))
                    prompt = data.get('prompt', '')
                    msgs = prefix_messages[:] + [{"role": "user", "content": prompt}]
                    reply = _run_tool_loop(msgs, available_tools, tool_map, opts)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(json.dumps({"response": reply}, ensure_ascii=False).encode('utf-8'))
                else:
                    self.send_response(404)
                    self.end_headers()
        server_address = ('', 8080)
        httpd = HTTPServer(server_address, AarisAPI)
        console.print("[bold green]Starting daemon server on port 8080...[/bold green]")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        return

    if "--run-prompt" in sys.argv:
        idx = sys.argv.index("--run-prompt")
        if idx + 1 < len(sys.argv):
            prompt = sys.argv[idx + 1]
            msgs = prefix_messages[:] + [{"role": "user", "content": prompt}]
            reply = _run_tool_loop(msgs, available_tools, tool_map, opts)
            console.print(Markdown(reply))
        return

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]Tú[/bold green]")
            if user_input.lower() in ("salir", "exit", "quit"):
                console.print("[bold yellow]¡Hasta luego![/bold yellow]")
                break

            if not user_input.strip():
                continue

            low = user_input.lower().strip()
            if low in ("reset memoria", "olvidar memoria", "borrar memoria"):
                memory = {
                    "memory_summary": "",
                    "stable_facts": [],
                    "preferences": {},
                    "last_updated": "",
                    "last_turns": [],
                }
                _save_memory(memory_path, memory)
                messages = _build_prefix_messages(memory)
                console.print("[dim]Memoria reiniciada. Empiezas con contexto limpio.[/dim]")
                continue

            if low == "ver memoria":
                console.print("\n[bold purple]Memoria persistente:[/bold purple]")
                console.print(Markdown(memory.get("memory_summary") or "(vacía)"))
                stable_facts = memory.get("stable_facts") or []
                if stable_facts:
                    console.print("\n[dim]Hechos estables:[/dim]")
                    for fact in stable_facts[:20]:
                        console.print(f"- {fact}")
                continue

            if low in ("undo last", "undo último", "undo ultimo"):
                try:
                    state = _load_undo_redo_state()
                    if not state.get("undo"):
                        console.print("[dim]Nada para deshacer.[/dim]")
                        continue
                    entry = state["undo"].pop()
                    state.setdefault("redo", []).append(entry)
                    _save_undo_redo_state(state)
                    tokens = entry.get("rollback_tokens") or []
                    tokens_str = ",".join(tokens)
                    if not tokens_str:
                        console.print("[dim]El último undo no tiene tokens de rollback guardados.[/dim]")
                        continue
                    if DRY_RUN:
                        console.print("[dim]DRY_RUN activo; no ejecuto undo.[/dim]")
                        continue
                    rb_res = rollback_tokens(tokens_str, overwrite=True)
                    console.print(f"\n[bold purple]Undo:[/bold purple] {rb_res}")
                except Exception as e:
                    console.print(f"[bold red]Error en undo:[/bold red] {e}")
                continue

            if low in ("redo last", "redo último", "redo ultimo"):
                try:
                    state = _load_undo_redo_state()
                    if not state.get("redo"):
                        console.print("[dim]Nada para rehacer.[/dim]")
                        continue
                    entry = state["redo"].pop()
                    state.setdefault("undo", []).append(entry)
                    _save_undo_redo_state(state)
                    tool_calls = entry.get("tool_calls") or []
                    if not tool_calls:
                        console.print("[dim]El último redo no tiene tool_calls guardados.[/dim]")
                        continue
                    if DRY_RUN:
                        console.print("[dim]DRY_RUN activo; no ejecuto redo.[/dim]")
                        continue
                    console.print(f"\n[bold purple]Redo ejecutando {len(tool_calls)} tools:[/bold purple]")
                    for tc in tool_calls:
                        name = tc.get("name") or ""
                        args = tc.get("arguments") or {}
                        if name and name in tool_map:
                            if name == "delete_path" and bool(args.get("recursive")):
                                args["confirm"] = True
                            if name in ("apply_unified_patch", "install_packages", "service_restart", "service_restart_with_deps"):
                                if "confirm" in args and not args.get("confirm"):
                                    args["confirm"] = True
                            func = tool_map[name]
                            res = func(**args)
                            console.print(f"[dim green]✓ {name}[/dim green] {str(res)[:200]}")
                except Exception as e:
                    console.print(f"[bold red]Error en redo:[/bold red] {e}")
                continue

            if low in ("history last", "history ultimo", "historial ultimo"):
                try:
                    if not os.path.isfile(log_path):
                        console.print("[dim]No hay logs.[/dim]")
                        continue
                    lines = []
                    with open(log_path, "r", encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            lines.append(line)
                    tail = lines[-10:]
                    for line in tail:
                        obj = json.loads(line)
                        ts = obj.get("ts")
                        user = obj.get("user")
                        tool_calls = obj.get("tool_calls") or []
                        tool_names = [tc.get("name") for tc in tool_calls]
                        console.print(f"- {ts} | {user} | tools={tool_names}")
                except Exception as e:
                    console.print(f"[bold red]Error en history:[/bold red] {e}")
                continue

            if low.startswith("history search "):
                term = user_input[len("history search ") :].strip()
                if not term:
                    console.print("[dim]Uso: history search <texto>[/dim]")
                    continue
                try:
                    if not os.path.isfile(log_path):
                        console.print("[dim]No hay logs.[/dim]")
                        continue
                    matches = 0
                    with open(log_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if term.lower() in line.lower():
                                matches += 1
                                if matches <= 10:
                                    obj = json.loads(line)
                                    ts = obj.get("ts")
                                    user = obj.get("user")
                                    tool_calls = obj.get("tool_calls") or []
                                    tool_names = [tc.get("name") for tc in tool_calls]
                                    console.print(f"- {ts} | {user} | tools={tool_names}")
                    if matches == 0:
                        console.print("[dim]Sin coincidencias.[/dim]")
                except Exception as e:
                    console.print(f"[bold red]Error en history search:[/bold red] {e}")
                continue

            if low in ("capabilities", "cap", "capacidades"):
                console.print("[bold purple]Capabilities:[/bold purple]")
                console.print(f"- AARIS_DRY_RUN={DRY_RUN}")
                console.print(f"- AARIS_PLAN_MODE={PLAN_MODE}")
                console.print(f"- AARIS_READ_ONLY={os.environ.get('AARIS_READ_ONLY', 'false')}")
                console.print(f"- AARIS_USE_TRASH={os.environ.get('AARIS_USE_TRASH', 'true')}")
                console.print(f"- AARIS_COMMAND_SANDBOX={os.environ.get('AARIS_COMMAND_SANDBOX', '(none)')}")
                console.print(f"- Tools disponibles={len(available_tools)}")
                continue

            if low in ("workspace show", "workspace", "ver workspace"):
                prefs = memory.get("preferences") or {}
                ws_root = prefs.get("workspace_root") if isinstance(prefs, dict) else None
                console.print(
                    f"[bold purple]Workspace:[/bold purple] cwd={os.getcwd()}\n"
                    f"workspace_root={ws_root or '(no configurado)'}"
                )
                continue

            if low.startswith("set workspace "):
                raw = user_input[len("set workspace ") :].strip()
                resolved_ws = resolve_path(raw, must_exist=True)
                if str(resolved_ws).startswith("Error:"):
                    console.print(f"[bold red]Error:[/bold red] {resolved_ws}")
                    continue
                if not isinstance(memory.get("preferences"), dict):
                    memory["preferences"] = {}
                memory["preferences"]["workspace_root"] = resolved_ws
                _save_memory(memory_path, memory)
                os.chdir(resolved_ws)
                messages = _build_prefix_messages(memory)
                console.print(f"[dim]Workspace fijado: {resolved_ws}[/dim]")
                continue

            if low in ("reset workspace", "clear workspace", "unset workspace"):
                if isinstance(memory.get("preferences"), dict):
                    memory["preferences"].pop("workspace_root", None)
                    _save_memory(memory_path, memory)
                os.chdir(str(Path.home()))
                messages = _build_prefix_messages(memory)
                console.print("[dim]Workspace eliminado. Vuelves a tu home.[/dim]")
                continue

            if low in ("rollback last", "undo last", "deshacer last", "rollback último", "undo último"):
                try:
                    if not os.path.isfile(log_path):
                        console.print("[dim]No hay logs para rollback todavía.[/dim]")
                        continue
                    last_token = None
                    with open(log_path, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            tool_calls = obj.get("tool_calls") or []
                            for tc in tool_calls:
                                rp = tc.get("result_preview") or ""
                                if "ROLLBACK_TOKEN=" in rp:
                                    m = re.search(r"ROLLBACK_TOKEN=([a-fA-F0-9]+)", rp)
                                    if m:
                                        last_token = m.group(1)
                                if "ROLLBACK_TOKENS=" in rp:
                                    m2 = re.search(r"ROLLBACK_TOKENS=([a-zA-Z0-9,-]+)", rp)
                                    if m2:
                                        parts = [x for x in m2.group(1).split(",") if x.strip()]
                                        if parts:
                                            last_token = parts[-1]
                    if not last_token:
                        console.print("[dim]No encontré tokens de rollback en el último log.[/dim]")
                        continue

                    ans_overwrite = Prompt.ask("Si el destino existe, ¿lo sobreescribo? (s/N)", default="N").strip().lower()
                    overwrite = ans_overwrite in ("s", "si", "sí", "y", "yes")
                    res = rollback(last_token, overwrite=overwrite)
                    console.print(f"\n[bold purple]Rollback:[/bold purple] {res}")
                    continue
                except Exception as e:
                    console.print(f"[bold red]Error en rollback:[/bold red] {e}")
                    continue

            if low.startswith("rollback "):
                token = low.split(" ", 1)[1].strip()
                if token:
                    overwrite = False
                    ans_overwrite = Prompt.ask("¿Sobreescribir si existe? (s/N)", default="N").strip().lower()
                    overwrite = ans_overwrite in ("s", "si", "sí", "y", "yes")
                    res = rollback(token, overwrite=overwrite)
                    console.print(f"\n[bold purple]Rollback:[/bold purple] {res}")
                continue

            if low in ("replay ultimo", "replay last", "replay último", "replay"):
                last_line = None
                try:
                    if os.path.isfile(log_path):
                        with open(log_path, "r", encoding="utf-8") as f:
                            for line in f:
                                last_line = line
                    if not last_line:
                        console.print("[dim]No hay logs para reproducir todavía.[/dim]")
                        continue

                    last_log = json.loads(last_line)
                    tool_calls = last_log.get("tool_calls") or []
                    if not tool_calls:
                        console.print("[dim]El último log no tiene tool_calls.[/dim]")
                        continue

                    console.print(f"\n[bold]Replaying último (tool_calls={len(tool_calls)})[/bold]")
                    if DRY_RUN:
                        console.print("[dim]DRY_RUN está activo; no ejecuto durante replay.[/dim]")
                        for tc in tool_calls:
                            console.print(f"- {tc.get('name')}: {tc.get('arguments')}")
                        continue

                    ans = Prompt.ask("¿Ejecutar exactamente esas herramientas? (s/N)", default="N").strip().lower()
                    if ans not in ("s", "si", "sí", "y", "yes"):
                        console.print("[dim]Replay cancelado.[/dim]")
                        continue

                    for tc in tool_calls:
                        name = tc.get("name") or ""
                        args = tc.get("arguments") or {}
                        if name and name in tool_map:
                            if name == "delete_path" and bool(args.get("recursive")) and not args.get("confirm"):
                                args["confirm"] = True
                            func = tool_map[name]
                            res = func(**args)
                            console.print(f"[dim green]✓ {name}[/dim green] {res[:300]}")
                        else:
                            console.print(f"[dim red]Herramienta desconocida en log:[/dim red] {name}")
                    continue
                except Exception as e:
                    console.print(f"[bold red]Error en replay:[/bold red] {e}")
                    continue

            plan_info: dict[str, Any] | None = None
            if PLAN_MODE in ("auto", "confirm"):
                plan_info = _plan_turn(user_input, messages, opts)
                if plan_info.get("requires_tools"):
                    console.print("\n[bold]Plan:[/bold]")
                    if plan_info.get("summary"):
                        console.print(Markdown(str(plan_info.get("summary"))))
                    for step in plan_info.get("steps") or []:
                        console.print(f"- {step}")
                    for note in plan_info.get("safety_notes") or []:
                        console.print(f"[dim]Seguridad: {note}[/dim]")

                    if PLAN_MODE == "confirm":
                        ans = Prompt.ask("¿Ejecutar ahora? (s/N)", default="N").strip().lower()
                        if ans not in ("s", "si", "sí", "y", "yes"):
                            console.print("[dim]Ejecución cancelada. Te dejo el plan listo.[/dim]")
                            continue

            # Mensaje al "worker" para seguir el plan (rol OpenClaw-like).
            if plan_info and plan_info.get("requires_tools"):
                steps = plan_info.get("steps") or []
                summary = plan_info.get("summary") or ""
                steps_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps[:8])]) if steps else "(sin pasos explícitos)"
                messages.append(
                    {
                        "role": "system",
                        "content": "Eres el worker. Sigue el plan descrito para cumplir la tarea. "
                        f"Resumen del plan: {summary}\nPasos:\n{steps_text}\n"
                        "Usa herramientas solo para ejecutar acciones del plan y detente cuando el plan esté completo; "
                        "si falta algo del plan, pide aclaración.",
                    }
                )

            turn_start_idx = len(messages)
            messages.append({"role": "user", "content": user_input})

            turn_tool_start = datetime.now().timestamp()
            reply_content = _run_tool_loop(messages, available_tools, tool_map, opts)
            turn_tool_ms = int((datetime.now().timestamp() - turn_tool_start) * 1000)

            if reply_content.strip():
                console.print("\n[bold purple]Asistente:[/bold purple]")
                console.print(Markdown(reply_content))

            # Log simple tipo "Open Claw": usuario + tools usados + salida truncada.
            try:
                tool_calls_extracted: list[dict[str, Any]] = []
                tool_results: list[str] = []
                for m in messages[turn_start_idx:]:
                    if m.get("role") == "assistant" and m.get("tool_calls"):
                        for tc in m.get("tool_calls") or []:
                            fn = (tc.get("function") or {})
                            name = fn.get("name") or ""
                            args = _normalize_tool_arguments(fn.get("arguments"))
                            if name:
                                tool_calls_extracted.append({"name": name, "arguments": args})
                    if m.get("role") == "tool":
                        # El resultado de la tool está en content.
                        tool_results.append(m.get("content") or "")

                tool_calls_log = []
                for i, tc in enumerate(tool_calls_extracted):
                    res_preview = tool_results[i][:800] if i < len(tool_results) else ""
                    tool_calls_log.append(
                        {"name": tc["name"], "arguments": tc["arguments"], "result_preview": res_preview}
                    )
                log_obj = {
                    "ts": _now_iso(),
                    "user": user_input,
                    "plan_mode": PLAN_MODE,
                    "plan": plan_info,
                    "tool_calls": tool_calls_log,
                    "tool_call_count": len(tool_calls_log),
                    "tool_loop_ms": turn_tool_ms,
                    "assistant_preview": reply_content[:1200],
                }
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_obj, ensure_ascii=False) + "\n")

                # Guardamos estado para undo/redo si hay tokens de rollback.
                try:
                    tokens_found: list[str] = []
                    for tr in tool_results:
                        if not isinstance(tr, str):
                            continue
                        mt = re.findall(r"ROLLBACK_TOKEN=([a-fA-F0-9]+)", tr)
                        tokens_found.extend(mt)
                        mt2 = re.findall(r"ROLLBACK_TOKENS=([a-zA-Z0-9,-]+)", tr)
                        for group in mt2:
                            tokens_found.extend([x for x in group.split(",") if x.strip()])
                    tokens_found = [t for t in tokens_found if t]
                    if tokens_found and tool_calls_log:
                        state = _load_undo_redo_state()
                        state.setdefault("undo", []).append(
                            {
                                "ts": log_obj.get("ts"),
                                "tool_calls": tool_calls_log,
                                "rollback_tokens": list(dict.fromkeys(tokens_found))[-50:],
                            }
                        )
                        state["redo"] = []
                        _save_undo_redo_state(state)
                except Exception:
                    pass
            except Exception:
                pass

            # Mantener contexto controlado para no inflar la ventana.
            messages = _prune_messages(messages, keep_last=MAX_CONTEXT_MESSAGES)

            # Persistir memoria entre sesiones (si no se reinició).
            memory = _update_memory(messages, memory, opts)
            _save_memory(memory_path, memory)

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Cancelado. Saliendo…[/bold yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
            messages.append({"role": "system", "content": f"Error previo (no repetir): {e}"})


if __name__ == "__main__":
    main()
