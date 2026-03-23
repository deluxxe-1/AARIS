import glob as glob_module
import json
import os
import subprocess
from pathlib import Path
import re
import difflib
import shutil
import unicodedata
from typing import Optional, Any
import fnmatch
import uuid
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import time
from contextlib import contextmanager


def _norm_text(s: str) -> str:
    # Normaliza acentos y hace casefold para comparaciones robustas.
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.casefold()


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def _read_only_mode() -> bool:
    val = os.environ.get("AARIS_READ_ONLY", "false").strip().lower()
    return val in ("1", "true", "yes", "si", "sí", "on")


def _read_only_allow_undo() -> bool:
    val = os.environ.get("AARIS_READ_ONLY_ALLOW_UNDO", "true").strip().lower()
    return val in ("1", "true", "yes", "si", "sí", "on")


_POLICY_CACHE: Optional[dict[str, Any]] = None


def _load_policy() -> dict[str, Any]:
    """
    Carga políticas desde env `AARIS_POLICY_JSON` (string JSON).
    Ejemplo:
      export AARIS_POLICY_JSON='{"forbidden_path_prefixes":["/etc/"],"require_confirm_tools":["delete_path"]}'
    """
    global _POLICY_CACHE
    if _POLICY_CACHE is not None:
        return _POLICY_CACHE
    raw = os.environ.get("AARIS_POLICY_JSON", "").strip()
    if not raw:
        # Si no hay env, intentamos con fichero.
        policy_path = os.environ.get(
            "AARIS_POLICY_PATH",
            os.path.join(os.path.expanduser("~"), ".aaris", "policy.json"),
        )
        try:
            p = Path(policy_path).expanduser()
            if p.is_file():
                raw = p.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            raw = ""

    if not raw:
        _POLICY_CACHE = {}
        return _POLICY_CACHE
    try:
        parsed = json.loads(raw)
        _POLICY_CACHE = parsed if isinstance(parsed, dict) else {}
    except Exception:
        _POLICY_CACHE = {}
    return _POLICY_CACHE


def _policy_forbidden_reason(abs_path: str) -> Optional[str]:
    policy = _load_policy()
    forbidden = policy.get("forbidden_path_prefixes") or []
    if not isinstance(forbidden, list):
        return None
    for pref in forbidden:
        if not isinstance(pref, str):
            continue
        if pref and abs_path.startswith(pref):
            return f"Error: política prohíbe tocar {pref}. Ruta: {abs_path}"
    return None


def _policy_require_confirm(tool_name: str) -> bool:
    policy = _load_policy()
    req = policy.get("require_confirm_tools") or []
    if not isinstance(req, list):
        return False
    return tool_name in req


def policy_show() -> str:
    """
    Devuelve la política actual (JSON) usada por el agente.
    """
    try:
        return json.dumps(_load_policy(), ensure_ascii=False)
    except Exception as e:
        return f"Error en policy_show: {e}"


def policy_set(policy_json: str, allow_dangerous: bool = False) -> str:
    """
    Sobrescribe la política en el fichero AARIS_POLICY_PATH.

    Requiere allow_dangerous=true (esta tool escribe disco).
    """
    try:
        if not allow_dangerous:
            return "Error: policy_set escribe disco. Usa allow_dangerous=true."
        if not policy_json or not policy_json.strip():
            return "Error: policy_json vacío."
        parsed = json.loads(policy_json)
        if not isinstance(parsed, dict):
            return "Error: policy_json debe ser un objeto JSON."
        policy_path = os.environ.get(
            "AARIS_POLICY_PATH",
            os.path.join(os.path.expanduser("~"), ".aaris", "policy.json"),
        )
        p = Path(policy_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
        # Limpia cache para recargar.
        global _POLICY_CACHE
        _POLICY_CACHE = None
        return f"OK policy_set en {p}"
    except Exception as e:
        return f"Error en policy_set: {e}"


def policy_reset(allow_dangerous: bool = False) -> str:
    """
    Resetea la política borrando el fichero de política (si existe).
    """
    try:
        if not allow_dangerous:
            return "Error: policy_reset escribe/borrado disco. Usa allow_dangerous=true."
        policy_path = os.environ.get(
            "AARIS_POLICY_PATH",
            os.path.join(os.path.expanduser("~"), ".aaris", "policy.json"),
        )
        p = Path(policy_path).expanduser()
        if p.is_file():
            p.unlink()
        global _POLICY_CACHE
        _POLICY_CACHE = None
        return f"OK policy_reset ({p})"
    except Exception as e:
        return f"Error en policy_reset: {e}"


def _allow_symlink_escape() -> bool:
    val = os.environ.get("AARIS_ALLOW_SYMLINK_ESCAPE", "false").strip().lower()
    return val in ("1", "true", "yes", "si", "sí", "on")


def _symlink_escapes_home(abs_path: str) -> bool:
    try:
        home = Path.home().resolve()
        p = Path(abs_path)
        if not p.is_symlink():
            return False
        target = p.resolve()
        return not _is_relative_to(target, home)
    except Exception:
        return True


def _validate_symlink_for_path(abs_path: str, allow_escape: bool = False) -> Optional[str]:
    if allow_escape:
        return None
    try:
        home = Path.home().resolve()
        p = Path(abs_path)
        # Recorremos componentes y bloqueamos si cualquier componente es symlink
        # que resuelva fuera del home (deep escape).
        cur = Path(p.root)
        for part in p.parts[1:]:
            cur = cur / part
            if cur.is_symlink():
                try:
                    target = cur.resolve()
                except Exception:
                    return f"Error: no pude resolver symlink: {cur}"
                if not _is_relative_to(target, home):
                    return f"Error: un componente es symlink y escapa del home. Ruta: {abs_path} (via {cur} -> {target})"
        # Si el path final también es symlink, mantenemos el check.
        if p.is_symlink() and _symlink_escapes_home(abs_path):
            return f"Error: la ruta es un symlink y apunta fuera de tu home. Ruta: {abs_path}"
        return None
    except Exception:
        # Si algo falla en el check, preferimos permitir para no romper flujos,
        # pero podrías endurecer si lo quieres.
        return None


def _backup_base_dir() -> Path:
    # Por defecto intentamos guardar backups dentro del cwd para compatibilidad con sandboxes.
    # El usuario puede forzar otra ubicación con `AARIS_BACKUP_PATH`.
    if os.environ.get("AARIS_BACKUP_PATH"):
        return Path(os.environ.get("AARIS_BACKUP_PATH", "")).expanduser().resolve()
    app_dir = os.environ.get("AARIS_APP_DIR")
    if app_dir:
        return Path(app_dir).expanduser().resolve() / "backups"
    return Path(os.getcwd()).resolve() / ".aaris" / "backups"


def _ensure_backup_dirs() -> tuple[Path, Path]:
    base = _backup_base_dir()
    files_dir = base / "files"
    meta_dir = base / "meta"
    files_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    return files_dir, meta_dir


def _lock_base_dir() -> Path:
    # Prefer a path inside the current working directory for sandbox compatibility.
    if os.environ.get("AARIS_LOCK_PATH"):
        return Path(os.environ.get("AARIS_LOCK_PATH", "")).expanduser().resolve()
    return Path(os.getcwd()).resolve() / ".aaris" / "locks"


def _acquire_path_lock(abs_path: str, timeout_seconds: int = 25) -> Optional[Path]:
    """
    Bloqueo simple por ruta (archivo lock) para evitar condiciones de carrera.
    """
    try:
        lock_dir = _lock_base_dir()
        lock_dir.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256(abs_path.encode("utf-8", errors="replace")).hexdigest()
        lock_file = lock_dir / f"{h}.lock"
        start = time.time()
        while True:
            try:
                fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, b"lock")
                os.close(fd)
                return lock_file
            except FileExistsError:
                if (time.time() - start) >= timeout_seconds:
                    return None
                time.sleep(0.1)
    except Exception:
        return None


def _release_path_lock(lock_file: Path) -> None:
    try:
        if lock_file.exists():
            lock_file.unlink()
    except Exception:
        pass


@contextmanager
def _path_lock(abs_path: str):
    lock_file = _acquire_path_lock(abs_path)
    if lock_file is None:
        raise RuntimeError("No pude adquirir lock de ruta (timeout).")
    try:
        yield
    finally:
        _release_path_lock(lock_file)


def _create_file_backup(src: Path) -> str:
    """
    Crea un backup de un archivo existente para rollback.
    Retorna un token.
    """
    files_dir, meta_dir = _ensure_backup_dirs()
    token = uuid.uuid4().hex
    backup_path = files_dir / f"{token}.bak"

    # Copiamos bytes para soportar cualquier contenido.
    backup_path.write_bytes(src.read_bytes())
    meta_path = meta_dir / f"{token}.json"
    meta = {
        "type": "file",
        "original_path": str(src.resolve()),
        "backup_path": str(backup_path),
        "created_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return token


def _create_move_backup(from_path: Path, to_path: Path) -> str:
    files_dir, meta_dir = _ensure_backup_dirs()
    token = uuid.uuid4().hex
    _ = files_dir  # solo para asegurar dirs
    meta_path = meta_dir / f"{token}.json"
    meta = {
        "type": "move",
        "from_path": str(from_path.resolve()),
        "to_path": str(to_path.resolve()),
        "created_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return token


def _create_trash_backup(original_path: Path, trash_dest: Path) -> str:
    files_dir, meta_dir = _ensure_backup_dirs()
    _ = files_dir
    token = uuid.uuid4().hex
    meta_path = meta_dir / f"{token}.json"
    meta = {
        "type": "trash",
        "original_path": str(original_path.resolve()),
        "trash_dest": str(trash_dest.resolve()),
        "created_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return token


def rollback(token: str, overwrite: bool = False) -> str:
    """
    Restaura una acción respaldada por token (últimos cambios con backup).
    """
    try:
        if _read_only_mode() and not _read_only_allow_undo():
            return "Error: AARIS_READ_ONLY=true. rollback deshabilitado."
        base = _backup_base_dir()
        meta_path = base / "meta" / f"{token}.json"
        if not meta_path.is_file():
            return f"Error: token no existe: {token}"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        mtype = meta.get("type")

        if mtype == "file":
            original_path = Path(str(meta.get("original_path") or "")).expanduser()
            backup_path = Path(str(meta.get("backup_path") or "")).expanduser()
            if not backup_path.is_file():
                return f"Error: backup no existe: {backup_path}"
            if original_path.exists() and not overwrite:
                return f"Error: destino existe: {original_path}. Usa overwrite=true."
            original_path.parent.mkdir(parents=True, exist_ok=True)
            original_path.write_bytes(backup_path.read_bytes())
            return f"Rollback OK: restaurado {original_path}"

        if mtype == "move":
            from_path = Path(str(meta.get("from_path") or "")).expanduser()
            to_path = Path(str(meta.get("to_path") or "")).expanduser()
            if not to_path.exists():
                return f"Error: no existe la ruta donde estaba movido: {to_path}"
            if from_path.exists() and not overwrite:
                return f"Error: destino existe: {from_path}. Usa overwrite=true."
            from_path.parent.mkdir(parents=True, exist_ok=True)
            if from_path.exists() and overwrite:
                if from_path.is_dir():
                    shutil.rmtree(str(from_path))
                else:
                    from_path.unlink()
            shutil.move(str(to_path), str(from_path))
            return f"Rollback OK: movido de vuelta a {from_path}"

        if mtype == "trash":
            original_path = Path(str(meta.get("original_path") or "")).expanduser()
            trash_dest = Path(str(meta.get("trash_dest") or "")).expanduser()
            if not trash_dest.exists():
                return f"Error: la entrada de Trash no existe: {trash_dest}"
            if original_path.exists() and not overwrite:
                return f"Error: destino existe: {original_path}. Usa overwrite=true."
            original_path.parent.mkdir(parents=True, exist_ok=True)
            if original_path.exists() and overwrite:
                if original_path.is_dir():
                    shutil.rmtree(str(original_path))
                else:
                    original_path.unlink()
            shutil.move(str(trash_dest), str(original_path))
            return f"Rollback OK: restaurado desde Trash a {original_path}"

        return f"Error: tipo de token no soportado: {mtype}"
    except Exception as e:
        return f"Error en rollback: {e}"


def rollback_tokens(tokens: str, overwrite: bool = False) -> str:
    """
    Restaura múltiples tokens generados por borrados/ediciones (p.ej. Trash masivo).
    `tokens` puede ser una lista separada por comas o espacios.
    """
    try:
        if _read_only_mode() and not _read_only_allow_undo():
            return "Error: AARIS_READ_ONLY=true. rollback_tokens deshabilitado."
        raw = (tokens or "").strip()
        if not raw:
            return "Error: tokens vacío."
        parts = [p.strip() for p in raw.replace(";", ",").replace("|", ",").split(",") if p.strip()]
        # Si viene como "a b c" (sin comas), fallback.
        if len(parts) <= 1 and " " in raw and "," not in raw:
            parts = [p.strip() for p in raw.split() if p.strip()]

        results: list[str] = []
        for t in parts:
            results.append(f"{t}: {rollback(t, overwrite=overwrite)}")
        ok = sum(1 for r in results if "Rollback OK" in r)
        return json.dumps({"tokens": parts, "ok": ok, "results": results}, ensure_ascii=False)
    except Exception as e:
        return f"Error en rollback_tokens: {e}"


_XDG_USER_DIRS_CACHE: Optional[dict[str, str]] = None


def _load_xdg_user_dirs() -> dict[str, str]:
    """
    Lee `~/.config/user-dirs.dirs` (XDG user dirs) y devuelve un mapeo por categoría.

    Categorías soportadas: documents, downloads, desktop, music, pictures, videos, public.
    """
    global _XDG_USER_DIRS_CACHE
    if _XDG_USER_DIRS_CACHE is not None:
        return _XDG_USER_DIRS_CACHE

    home = Path.home().resolve()
    xdg_file = home / ".config" / "user-dirs.dirs"
    result: dict[str, str] = {}
    if not xdg_file.is_file():
        _XDG_USER_DIRS_CACHE = result
        return result

    var_map = {
        "documents": "XDG_DOCUMENTS_DIR",
        "downloads": "XDG_DOWNLOAD_DIR",
        "desktop": "XDG_DESKTOP_DIR",
        "music": "XDG_MUSIC_DIR",
        "pictures": "XDG_PICTURES_DIR",
        "videos": "XDG_VIDEOS_DIR",
        "public": "XDG_PUBLICSHARE_DIR",
    }

    try:
        text = xdg_file.read_text(encoding="utf-8", errors="replace")
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")

            # Solo nos interesan las variables del mapa.
            cat: Optional[str] = None
            for c, v in var_map.items():
                if v == key:
                    cat = c
                    break
            if not cat:
                continue

            # Sustituye $HOME y expande el resto de variables.
            if "$HOME" in val:
                val = val.replace("$HOME", str(home))
            expanded = os.path.expandvars(val)
            expanded = os.path.abspath(expanded)

            if expanded.startswith(str(home)):
                result[cat] = expanded
    except Exception:
        # Si hay corrupción/encoding raro, simplemente caemos en el mapeo fuzzy.
        result = {}

    _XDG_USER_DIRS_CACHE = result
    return result


def create_file(path: str, content: str = "") -> str:
    """
    Crea un archivo en la ruta indicada. Crea directorios padre si no existen.

    Args:
        path: Ruta del archivo (absoluta o relativa al directorio de trabajo).
        content: Contenido inicial del archivo (texto UTF-8).
    """
    try:
        if _read_only_mode():
            return "Error: AARIS_READ_ONLY=true. create_file deshabilitado."
        resolved = resolve_path(path, must_exist=False)
        if resolved.startswith("Error:"):
            return resolved
        abs_path = resolved
        if os.path.exists(abs_path) and Path(abs_path).is_symlink():
            sym_err = _validate_symlink_for_path(abs_path, allow_escape=_allow_symlink_escape())
            if sym_err:
                return sym_err
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with _path_lock(abs_path):
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(content)
        return f"Archivo creado correctamente en {abs_path}."
    except Exception as e:
        return f"Error al crear el archivo: {e}"


def read_file(path: str, max_chars: int = 80000) -> str:
    """
    Lee el contenido de un archivo de texto existente.

    Args:
        path: Ruta del archivo.
        max_chars: Máximo de caracteres a devolver (el resto se trunca).
    """
    try:
        if _read_only_mode():
            # lectura permitida aunque esté read-only; esta rama se usa solo para mutaciones,
            # pero mantenemos compatibilidad si el usuario lo activa.
            pass
        resolved = resolve_path(path, must_exist=True)
        if resolved.startswith("Error:"):
            return resolved
        abs_path = resolved
        if not os.path.isfile(abs_path):
            return f"Error: no existe el archivo {abs_path}."
        sym_err = _validate_symlink_for_path(abs_path, allow_escape=_allow_symlink_escape())
        if sym_err:
            return sym_err
        data = _read_file_cached(abs_path, max_chars)
        if len(data) > max_chars:
            return data[:max_chars] + f"\n\n[… truncado: más de {max_chars} caracteres; usa read_file con rango menor o edición parcial.]"
        return data if data else "(archivo vacío)"
    except Exception as e:
        return f"Error al leer: {e}"


@lru_cache(maxsize=64)
def _read_file_cached(abs_path: str, max_chars: int) -> str:
    with open(abs_path, encoding="utf-8", errors="replace") as f:
        data = f.read(max_chars + 1)
    # normalizamos para que el cache sea estable
    return data


def edit_file(path: str, new_content: str) -> str:
    """
    Sustituye por completo el contenido de un archivo existente.

    Args:
        path: Ruta del archivo.
        new_content: Nuevo contenido completo del archivo.
    """
    try:
        if _read_only_mode():
            return "Error: AARIS_READ_ONLY=true. edit_file deshabilitado."
        resolved = resolve_path(path, must_exist=True)
        if resolved.startswith("Error:"):
            return resolved
        abs_path = resolved
        if not os.path.exists(abs_path):
            return f"Error: el archivo {abs_path} no existe. Créalo antes con create_file."
        sym_err = _validate_symlink_for_path(abs_path, allow_escape=_allow_symlink_escape())
        if sym_err:
            return sym_err
        pol_reason = _policy_forbidden_reason(abs_path)
        if pol_reason:
            return pol_reason
        with _path_lock(abs_path):
            token = _create_file_backup(Path(abs_path))
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            if abs_path.lower().endswith(".py"):
                rb = validate_python_syntax(abs_path)
                if rb.startswith("Error"):
                    rollback_res = rollback(token, overwrite=True)
                    return f"Error: validación Python falló en {abs_path}. {rollback_res}"
        if abs_path.lower().endswith(".json"):
            try:
                json.loads(new_content)
            except Exception:
                rollback_res = rollback(token, overwrite=True)
                return f"Error: JSON inválido en {abs_path}. {rollback_res}"
        _read_file_cached.cache_clear()
        _exists_path_cached.cache_clear()
        return f"Archivo actualizado: {abs_path}. ROLLBACK_TOKEN={token}"
    except Exception as e:
        return f"Error al editar: {e}"


def search_replace_in_file(
    path: str,
    old_text: str,
    new_text: str,
    replace_all: bool = False,
) -> str:
    """
    Reemplaza una porción de texto dentro de un archivo (útil para archivos grandes).

    Args:
        path: Ruta del archivo.
        old_text: Fragmento exacto a buscar.
        new_text: Texto de sustitución.
        replace_all: Si True, reemplaza todas las ocurrencias; si False, solo la primera.
    """
    try:
        if _read_only_mode():
            return "Error: AARIS_READ_ONLY=true. search_replace_in_file deshabilitado."
        resolved = resolve_path(path, must_exist=True)
        if resolved.startswith("Error:"):
            return resolved
        abs_path = resolved
        if not os.path.isfile(abs_path):
            return f"Error: no existe el archivo {abs_path}."
        sym_err = _validate_symlink_for_path(abs_path, allow_escape=_allow_symlink_escape())
        if sym_err:
            return sym_err
        pol_reason = _policy_forbidden_reason(abs_path)
        if pol_reason:
            return pol_reason
        with _path_lock(abs_path):
            token = _create_file_backup(Path(abs_path))
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
            if old_text not in content:
                return "Error: old_text no aparece en el archivo (comprueba espacios y saltos de línea)."
            if replace_all:
                n = content.count(old_text)
                new_content = content.replace(old_text, new_text)
                msg = f"Reemplazadas {n} ocurrencias."
            else:
                new_content = content.replace(old_text, new_text, 1)
                msg = "Reemplazada la primera ocurrencia."
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            if abs_path.lower().endswith(".json"):
                try:
                    json.loads(new_content)
                except Exception:
                    rollback_res = rollback(token, overwrite=True)
                    return f"Error: JSON inválido en {abs_path}. {rollback_res}"
            if abs_path.lower().endswith(".py"):
                rb = validate_python_syntax(abs_path)
                if rb.startswith("Error"):
                    rollback_res = rollback(token, overwrite=True)
                    return f"Error: validación Python falló en {abs_path}. {rollback_res}"
            _read_file_cached.cache_clear()
            _exists_path_cached.cache_clear()
            return f"{msg} Archivo: {abs_path}. ROLLBACK_TOKEN={token}"
    except Exception as e:
        return f"Error en search_replace: {e}"


def create_folder(path: str) -> str:
    """
    Crea un directorio (y padres si hace falta).

    Args:
        path: Ruta del directorio a crear.
    """
    try:
        if _read_only_mode():
            return "Error: AARIS_READ_ONLY=true. create_folder deshabilitado."
        resolved = resolve_path(path, must_exist=False)
        if resolved.startswith("Error:"):
            return resolved
        abs_path = resolved
        if os.path.exists(abs_path) and Path(abs_path).is_symlink():
            sym_err = _validate_symlink_for_path(abs_path, allow_escape=_allow_symlink_escape())
            if sym_err:
                return sym_err
        os.makedirs(abs_path, exist_ok=True)
        return f"Carpeta lista: {abs_path}."
    except Exception as e:
        return f"Error al crear carpeta: {e}"


def list_directory(path: str = ".", show_hidden: bool = False) -> str:
    """
    Lista entradas en un directorio (nombre y tipo: archivo o directorio).

    Args:
        path: Ruta del directorio (por defecto el actual).
        show_hidden: Si True, incluye entradas cuyo nombre empieza por punto.
    """
    try:
        resolved = resolve_path(path, must_exist=True)
        if resolved.startswith("Error:"):
            return resolved
        root = Path(resolved).resolve()
        if not root.is_dir():
            return f"Error: {root} no es un directorio."
        lines = []
        for p in sorted(root.iterdir()):
            if not show_hidden and p.name.startswith("."):
                continue
            kind = "dir" if p.is_dir() else "file"
            lines.append(f"[{kind}] {p.name}")
        return "\n".join(lines) if lines else "(directorio vacío o sin entradas visibles)"
    except Exception as e:
        return f"Error al listar: {e}"


def glob_find(pattern: str, root: str = ".") -> str:
    """
    Busca rutas que coincidan con un patrón glob (ej: **/*.py, src/*.txt).

    Args:
        pattern: Patrón glob relativo a root.
        root: Directorio base para la búsqueda.
    """
    try:
        resolved_root = resolve_path(root, must_exist=True)
        if resolved_root.startswith("Error:"):
            return resolved_root
        base = Path(resolved_root).resolve()
        paths = sorted(
            Path(p).resolve()
            for p in glob_module.glob(str(base / pattern), recursive=True)
        )
        if not paths:
            return f"Sin coincidencias para {pattern!r} bajo {base}."
        text = "\n".join(str(p) for p in paths[:200])
        if len(paths) > 200:
            text += f"\n… y {len(paths) - 200} más."
        return text
    except Exception as e:
        return f"Error en glob: {e}"


def resolve_path(path: str, cwd: Optional[str] = None, must_exist: bool = False) -> str:
    """
    Resuelve rutas “humanas” (por ejemplo `Documents`/`Documentos`) a rutas absolutas reales.

    - Si el usuario pasa `Documents/...` se interpreta como `$HOME/Documents/...` (aunque el nombre real varíe).
    - Si el usuario pasa una ruta absoluta (`/etc/...`) se respeta tal cual.
    """
    try:
        if cwd is None:
            cwd = os.getcwd()

        raw = (path or "").strip()
        if not raw:
            return "Error: path vacío."

        expanded = os.path.expanduser(raw)
        if os.path.isabs(expanded):
            abs_path = os.path.abspath(expanded)
            if must_exist and not os.path.exists(abs_path):
                return f"Error: la ruta no existe: {abs_path}"
            return abs_path

        home = Path.home().resolve()
        tokens = [t for t in expanded.split("/") if t]
        if not tokens:
            return "Error: path inválido."

        # Mapeo de alias comunes al home del usuario.
        home_aliases: dict[str, list[str]] = {
            _norm_text("documents"): ["Documents"],
            _norm_text("documentos"): ["Documents"],
            _norm_text("docs"): ["Documents"],
            _norm_text("doc"): ["Documents"],
            _norm_text("downloads"): ["Downloads"],
            _norm_text("descargas"): ["Downloads"],
            _norm_text("download"): ["Downloads"],
            _norm_text("desktop"): ["Desktop"],
            _norm_text("escritorio"): ["Desktop"],
            _norm_text("music"): ["Music"],
            _norm_text("musica"): ["Music"],
            _norm_text("música"): ["Music"],
            _norm_text("pictures"): ["Pictures", "Photos", "Images"],
            _norm_text("photos"): ["Pictures", "Photos", "Images"],
            _norm_text("images"): ["Pictures", "Photos", "Images"],
            _norm_text("imagenes"): ["Pictures", "Photos", "Images"],
            _norm_text("imágenes"): ["Pictures", "Photos", "Images"],
            _norm_text("videos"): ["Videos"],
            _norm_text("video"): ["Videos"],
            _norm_text("public"): ["Public"],
            _norm_text("público"): ["Public"],
            _norm_text("publico"): ["Public"],
        }

        # Primero intentamos usar XDG user dirs (mapeo real del sistema).
        xdg = _load_xdg_user_dirs()
        alias_to_category: dict[str, str] = {
            _norm_text("documents"): "documents",
            _norm_text("documentos"): "documents",
            _norm_text("docs"): "documents",
            _norm_text("doc"): "documents",
            _norm_text("downloads"): "downloads",
            _norm_text("descargas"): "downloads",
            _norm_text("download"): "downloads",
            _norm_text("desktop"): "desktop",
            _norm_text("escritorio"): "desktop",
            _norm_text("music"): "music",
            _norm_text("musica"): "music",
            _norm_text("música"): "music",
            _norm_text("pictures"): "pictures",
            _norm_text("photos"): "pictures",
            _norm_text("images"): "pictures",
            _norm_text("imagenes"): "pictures",
            _norm_text("imágenes"): "pictures",
            _norm_text("videos"): "videos",
            _norm_text("video"): "videos",
            _norm_text("public"): "public",
            _norm_text("público"): "public",
            _norm_text("publico"): "public",
        }

        first_norm = _norm_text(tokens[0])
        if first_norm in alias_to_category:
            cat = alias_to_category[first_norm]
            base_xdg = xdg.get(cat)
            if base_xdg:
                rest = "/".join(tokens[1:])
                abs_path = os.path.abspath(os.path.join(base_xdg, rest)) if rest else os.path.abspath(base_xdg)
                if must_exist and not os.path.exists(abs_path):
                    return f"Error: la ruta no existe: {abs_path}"
                return abs_path

        if first_norm in home_aliases:
            expected = home_aliases[first_norm]
            entries = [p for p in home.iterdir() if p.is_dir()]

            # 1) Match exacto ignorando case/acento.
            for e in entries:
                for exp in expected:
                    if _norm_text(e.name) == _norm_text(exp):
                        base_dir = e
                        rest = "/".join(tokens[1:])
                        abs_path = str((base_dir / rest).resolve()) if rest else str(base_dir)
                        if must_exist and not os.path.exists(abs_path):
                            return f"Error: la ruta no existe: {abs_path}"
                        return abs_path

            # 2) Fuzzy match entre carpetas candidatas del home.
            expected_norm = [_norm_text(x) for x in expected]
            best_score = 0.0
            best_name: Optional[str] = None
            scored_entries: list[tuple[float, str]] = []
            for e in entries:
                name_norm = _norm_text(e.name)
                score = max(difflib.SequenceMatcher(None, name_norm, exp).ratio() for exp in expected_norm)
                if score > best_score:
                    best_score = score
                    best_name = e.name
                scored_entries.append((score, e.name))

            if best_name and best_score >= 0.67:
                base_dir = home / best_name
                rest = "/".join(tokens[1:])
                abs_path = str((base_dir / rest).resolve()) if rest else str(base_dir)
                if must_exist and not os.path.exists(abs_path):
                    return f"Error: la ruta no existe: {abs_path}"
                return abs_path

            if must_exist and scored_entries:
                scored_entries.sort(key=lambda x: x[0], reverse=True)
                top = scored_entries[:5]
                candidates_json = [
                    {
                        "name": name,
                        "score": score,
                        "path": str((home / name).resolve()),
                    }
                    for score, name in top
                ]
                candidates = ", ".join([f"{name}({score:.2f})" for score, name in top])
                return (
                    f"Error: mapeo ambiguo para `{tokens[0]}`. "
                    f"Candidatos en tu $HOME: {candidates}. "
                    f"CANDIDATES_JSON={json.dumps(candidates_json, ensure_ascii=False)}. "
                    "Repite usando el nombre exacto (o una ruta absoluta)."
                )

            # Si el usuario solo quiere crear, permitimos caer en el nombre canónico esperado.
            if not must_exist:
                base_dir = home / expected[0]
                rest = "/".join(tokens[1:])
                abs_path = os.path.abspath(str(base_dir / rest)) if rest else os.path.abspath(str(base_dir))
                return abs_path

            return (
                f"Error: no pude mapear `{tokens[0]}` a una carpeta real en tu `$HOME`. "
                "Usa una ruta con el nombre exacto o una ruta absoluta."
            )

        # Rutas relativas: se interpretan desde el cwd del proceso.
        abs_path = os.path.abspath(os.path.join(cwd, expanded))
        if must_exist and not os.path.exists(abs_path):
            return f"Error: la ruta no existe: {abs_path}"
        return abs_path
    except Exception as e:
        return f"Error en resolve_path: {e}"


def exists_path(path: str, cwd: Optional[str] = None) -> str:
    """
    Comprueba si existe una ruta (archivo o directorio) resolviendo alias tipo `Documents`.
    """
    try:
        resolved = resolve_path(path, cwd=cwd, must_exist=False)
        if resolved.startswith("Error:"):
            return resolved
        return "true" if _exists_path_cached(resolved) else "false"
    except Exception as e:
        return f"Error en exists_path: {e}"


@lru_cache(maxsize=128)
def _exists_path_cached(resolved_abs_path: str) -> bool:
    return Path(resolved_abs_path).exists()


def stat_path(path: str, cwd: Optional[str] = None) -> str:
    """
    Devuelve un resumen JSON con tipo/tamaño/tiempos de una ruta.
    """
    try:
        resolved = resolve_path(path, cwd=cwd, must_exist=False)
        if resolved.startswith("Error:"):
            return resolved
        return _stat_path_cached(resolved)
    except Exception as e:
        return f"Error en stat_path: {e}"


@lru_cache(maxsize=64)
def _stat_path_cached(resolved_abs_path: str) -> str:
    p = Path(resolved_abs_path)
    if not p.exists() and not p.is_symlink():
        return f"Error: no existe {p}"
    st = p.lstat()
    info = {
        "path": str(p),
        "exists": p.exists(),
        "is_dir": p.is_dir(),
        "is_file": p.is_file(),
        "is_symlink": p.is_symlink(),
        "size_bytes": st.st_size,
        "mtime_iso": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }
    return json.dumps(info, ensure_ascii=False)


def describe_path(path: str, cwd: Optional[str] = None) -> str:
    """
    Devuelve una descripción legible (rápida) de una ruta.
    """
    try:
        resolved = resolve_path(path, cwd=cwd, must_exist=False)
        if resolved.startswith("Error:"):
            return resolved
        return _describe_path_cached(resolved)
    except Exception as e:
        return f"Error en describe_path: {e}"


@lru_cache(maxsize=64)
def _describe_path_cached(resolved_abs_path: str) -> str:
    p = Path(resolved_abs_path)
    if not p.exists() and not p.is_symlink():
        return f"(no existe) {p}"
    st = p.lstat()
    kind = "dir" if p.is_dir() else "file" if p.is_file() else "other"
    hidden = p.name.startswith(".")
    mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"{kind} | hidden={hidden} | size={st.st_size} bytes | mtime={mtime} | {p}"


def estimate_dir(path: str, cwd: Optional[str] = None, max_entries: int = 5000) -> str:
    """
    Estima rápidamente el alcance de un directorio contando entradas inmediatas hasta `max_entries`.
    """
    try:
        resolved = resolve_path(path, cwd=cwd, must_exist=True)
        if resolved.startswith("Error:"):
            return resolved
        return _estimate_dir_cached(resolved, max_entries)
    except Exception as e:
        return f"Error en estimate_dir: {e}"


@lru_cache(maxsize=32)
def _estimate_dir_cached(resolved_abs_path: str, max_entries: int) -> str:
    p = Path(resolved_abs_path)
    if not p.is_dir():
        return f"Error: {p} no es un directorio."
    entries = 0
    files = 0
    dirs = 0
    hidden = 0
    bytes_sum = 0
    for child in p.iterdir():
        entries += 1
        if child.name.startswith("."):
            hidden += 1
        if child.is_dir():
            dirs += 1
        elif child.is_file() or child.is_symlink():
            files += 1
            try:
                bytes_sum += child.lstat().st_size
            except Exception:
                pass
        if entries >= max_entries:
            break
    more = "" if entries < max_entries else " (posible más, no contamos todo)"
    info = {
        "dir": str(p),
        "entries_counted": entries,
        "files_counted": files,
        "dirs_counted": dirs,
        "hidden_counted": hidden,
        "size_bytes_sum_files_counted": bytes_sum,
        "note": more.strip(),
    }
    return json.dumps(info, ensure_ascii=False)


def disk_usage(path: str = "~") -> str:
    """
    Devuelve el uso de disco para la ruta.
    """
    try:
        resolved = resolve_path(path, must_exist=False)
        if resolved.startswith("Error:"):
            return resolved
        p = Path(resolved).resolve()
        usage = shutil.disk_usage(str(p))
        info = {
            "path": str(p),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        }
        return json.dumps(info, ensure_ascii=False)
    except Exception as e:
        return f"Error en disk_usage: {e}"


def list_processes(limit: int = 30) -> str:
    """
    Lista procesos visibles desde /proc con pid y cmdline (limitado).
    """
    try:
        procs = []
        proc_dir = Path("/proc")
        for pid_dir in proc_dir.iterdir():
            if not pid_dir.name.isdigit():
                continue
            pid = int(pid_dir.name)
            cmdline_file = pid_dir / "cmdline"
            try:
                raw = cmdline_file.read_bytes()
                cmd = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
            except Exception:
                cmd = ""
            if not cmd:
                # fallback: comm
                try:
                    cmd = (pid_dir / "comm").read_text(encoding="utf-8", errors="replace").strip()
                except Exception:
                    cmd = ""
            procs.append({"pid": pid, "cmd": cmd[:300]})
            if len(procs) >= limit:
                break
        procs.sort(key=lambda x: x["pid"])
        return json.dumps({"count": len(procs), "processes": procs}, ensure_ascii=False)
    except Exception as e:
        return f"Error en list_processes: {e}"


def tail_file(path: str, lines: int = 200, max_bytes: int = 2_000_000) -> str:
    """
    Lee las últimas `lines` líneas de un archivo.
    """
    try:
        resolved = resolve_path(path, must_exist=True)
        if resolved.startswith("Error:"):
            return resolved
        p = Path(resolved)
        if not p.is_file():
            return f"Error: {p} no es un archivo."

        # Estrategia simple: buscamos desde el final hacia atrás.
        with p.open("rb") as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            back = min(file_size, max_bytes)
            f.seek(file_size - back, os.SEEK_SET)
            chunk = f.read(back)

        text = chunk.decode("utf-8", errors="replace")
        all_lines = text.splitlines()
        tail = all_lines[-lines:] if lines > 0 else all_lines
        return "\n".join(tail)
    except Exception as e:
        return f"Error en tail_file: {e}"


def count_dir_children_matches(
    path: str,
    glob_filter: str,
    show_hidden: bool = False,
    cwd: Optional[str] = None,
) -> str:
    """
    Cuenta coincidencias inmediatas (solo hijos de 1 nivel) para un glob_filter.
    Útil para previsualizar `delete_path(..., recursive=true, glob_filter=...)`.
    """
    try:
        resolved = resolve_path(path, cwd=cwd, must_exist=True)
        if resolved.startswith("Error:"):
            return resolved
        p = Path(resolved).resolve()
        if not p.is_dir():
            return f"Error: {p} no es un directorio."
        count = 0
        matched_names: list[str] = []
        for child in p.iterdir():
            if not show_hidden and child.name.startswith("."):
                continue
            if fnmatch.fnmatch(child.name, glob_filter):
                count += 1
                if len(matched_names) < 30:
                    matched_names.append(child.name)
        return json.dumps(
            {"dir": str(p), "glob_filter": glob_filter, "count": count, "matched_names": matched_names},
            ensure_ascii=False,
        )
    except Exception as e:
        return f"Error en count_dir_children_matches: {e}"


def validate_python_syntax(path: str) -> str:
    """
    Valida sintaxis Python ejecutando `python -m py_compile`.
    """
    try:
        resolved = resolve_path(path, must_exist=True)
        if resolved.startswith("Error:"):
            return resolved
        p = Path(resolved)
        if not p.is_file():
            return f"Error: {p} no es un archivo."
        import sys
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(p)],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ},
        )
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout or "").strip()
            return f"Error: validación py_compile falló: {msg[:12000]}"
        return f"OK validate_python_syntax: {p}"
    except Exception as e:
        return f"Error en validate_python_syntax: {e}"


def service_status(service_name: str, allow_dangerous: bool = False) -> str:
    """
    Consulta estado de un servicio con systemctl.
    """
    try:
        if not shutil.which("systemctl"):
            return "Error: no existe `systemctl`."
        if not service_name or "/" in service_name:
            return "Error: service_name inválido."
        proc = subprocess.run(
            ["systemctl", "status", service_name, "--no-pager"],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ},
        )
        out = (proc.stdout or "") + (("\n--- stderr ---\n" + proc.stderr) if proc.stderr else "")
        if len(out) > 12000:
            out = out[:12000] + "\n[…truncado…]"
        return out if out.strip() else "(sin salida)"
    except Exception as e:
        return f"Error en service_status: {e}"


def service_restart(
    service_name: str,
    reload: bool = False,
    confirm: bool = False,
    allow_dangerous: bool = False,
) -> str:
    """
    Reinicia un servicio. Requiere confirm=true.
    """
    try:
        if _read_only_mode():
            return "Error: AARIS_READ_ONLY=true. service_restart deshabilitado."
        if not shutil.which("systemctl"):
            return "Error: no existe `systemctl`."
        if not service_name or "/" in service_name:
            return "Error: service_name inválido."
        if not confirm and not allow_dangerous:
            return "Error: confirm=true requerido para service_restart."
        cmd = ["systemctl", "reload" if reload else "restart", service_name]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ},
        )
        out = (proc.stdout or "") + (("\n--- stderr ---\n" + proc.stderr) if proc.stderr else "")
        exit_note = f"\n[código de salida: {proc.returncode}]"
        return (out if out.strip() else "(sin salida)") + exit_note
    except Exception as e:
        return f"Error en service_restart: {e}"


def service_wait_active(
    service_name: str,
    timeout_seconds: int = 60,
    poll_interval_seconds: float = 1.0,
) -> str:
    """
    Espera a que el servicio esté activo (systemctl is-active).
    """
    try:
        if not shutil.which("systemctl"):
            return "Error: no existe `systemctl`."
        if not service_name or "/" in service_name:
            return "Error: service_name inválido."
        start = time.time()
        while True:
            proc = subprocess.run(
                ["systemctl", "is-active", service_name],
                capture_output=True,
                text=True,
                timeout=10,
                env={**os.environ},
            )
            out = (proc.stdout or "").strip()
            if out == "active":
                health = service_health_report(service_name)
                return f"OK service_wait_active: {service_name} está activo.\nHealth: {health}"
            if time.time() - start >= timeout_seconds:
                return f"Error: timeout esperando active. Estado actual: {out or '(desconocido)'}"
            time.sleep(poll_interval_seconds)
    except Exception as e:
        return f"Error en service_wait_active: {e}"


def service_health_report(service_name: str, allow_dangerous: bool = False) -> str:
    """
    Devuelve un resumen estructurado del estado del servicio.
    """
    try:
        if not shutil.which("systemctl"):
            return "Error: no existe `systemctl`."
        if not service_name or "/" in service_name:
            return "Error: service_name inválido."
        proc = subprocess.run(
            [
                "systemctl",
                "show",
                service_name,
                "--property=ActiveState,SubState,Result,ExecMainStatus,NRestarts,MainPID",
                "--no-page",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            env={**os.environ},
        )
        out = (proc.stdout or "") + (("\n--- stderr ---\n" + proc.stderr) if proc.stderr else "")
        if proc.returncode != 0:
            return "Error en service_health_report:\n" + out[:12000]

        report: dict[str, Any] = {}
        for line in (proc.stdout or "").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                report[k.strip()] = v.strip()
        if not report:
            return "Error: no pude leer propiedades de systemctl."
        return json.dumps({"service": service_name, "health": report, "raw": out[:2000]}, ensure_ascii=False)
    except Exception as e:
        return f"Error en service_health_report: {e}"


def service_restart_with_deps(
    service_name: str,
    confirm: bool = False,
    allow_dangerous: bool = False,
    depth: int = 1,
) -> str:
    """
    Reinicia un servicio y sus dependencias más cercanas (1 nivel por defecto).
    """
    try:
        if _read_only_mode():
            return "Error: AARIS_READ_ONLY=true. service_restart_with_deps deshabilitado."
        if not shutil.which("systemctl"):
            return "Error: no existe `systemctl`."
        if not service_name or "/" in service_name:
            return "Error: service_name inválido."
        if not confirm and not allow_dangerous:
            return "Error: confirm=true requerido para service_restart_with_deps."

        # Solo consideramos units .service para evitar reinicios raros.
        def _get_requires(prop: str) -> list[str]:
            proc = subprocess.run(
                ["systemctl", "show", service_name, f"-p{prop}", "--value", "--no-pager"],
                capture_output=True,
                text=True,
                timeout=20,
                env={**os.environ},
            )
            if proc.returncode != 0:
                return []
            raw = (proc.stdout or "").strip()
            if not raw:
                return []
            units = [u.strip() for u in raw.replace(",", " ").split() if u.strip()]
            return [u for u in units if u.endswith(".service")]

        requires = _get_requires("Requires")
        after = _get_requires("After")
        deps = []
        for u in requires + after:
            if u not in deps and u != service_name:
                deps.append(u)

        # 1 nivel: reiniciamos primero dependencias.
        actions = []
        for d in deps[:50]:
            r = service_restart(d, reload=False, confirm=True, allow_dangerous=allow_dangerous)
            actions.append(f"{d}: {r.splitlines()[0][:200]}")

        main_res = service_restart(service_name, reload=False, confirm=True, allow_dangerous=allow_dangerous)
        return "OK service_restart_with_deps.\nDeps:\n- " + "\n- ".join(actions) + f"\nMain:\n{main_res[:12000]}"
    except Exception as e:
        return f"Error en service_restart_with_deps: {e}"


def detect_project(root: str = ".") -> str:
    """
    Detecta el tipo de proyecto por marcadores en el filesystem.
    """
    try:
        resolved_root = resolve_path(root, must_exist=True)
        if resolved_root.startswith("Error:"):
            return resolved_root
        r = Path(resolved_root).resolve()
        markers = []

        def _has(name: str) -> bool:
            return (r / name).exists()

        project_type = "unknown"
        if _has("package.json"):
            project_type = "node"
            markers.append("package.json")
        if _has("pyproject.toml") or _has("requirements.txt"):
            project_type = "python"
            markers.extend([m for m in ["pyproject.toml", "requirements.txt"] if _has(m)])
        if _has("Cargo.toml"):
            project_type = "rust"
            markers.append("Cargo.toml")
        if _has("go.mod"):
            project_type = "go"
            markers.append("go.mod")
        if _has("docker-compose.yml") or _has("Dockerfile"):
            markers.extend([m for m in ["docker-compose.yml", "Dockerfile"] if _has(m)])

        info = {"root": str(r), "type": project_type, "markers": sorted(set(markers))}
        return json.dumps(info, ensure_ascii=False)
    except Exception as e:
        return f"Error en detect_project: {e}"


def project_workflow_suggest(root: str = ".", include_commands: bool = True) -> str:
    """
    Sugiere un pipeline típico según el tipo de proyecto.
    No ejecuta nada: solo devuelve pasos recomendados para que el agente los orqueste.
    """
    try:
        info_raw = detect_project(root)
        if not info_raw.startswith("{"):
            return info_raw
        info = json.loads(info_raw)
        ptype = info.get("type") or "unknown"
        root_resolved = info.get("root") or root

        steps: list[str] = []
        cmds: list[str] = []

        if ptype == "python":
            steps = [
                "Instalar dependencias con el gestor indicado por el repo (requirements.txt/pyproject).",
                "Ejecutar tests (pytest/unittest) y capturar resultados.",
                "Validar sintaxis Python (compileall/py_compile).",
                "Revisar linting si existe (ruff/flake8).",
                "Aplicar cambios y verificar de nuevo.",
            ]
            cmds = [
                "python -m compileall .",
                "python -m pytest -q",
            ]
        elif ptype == "node":
            steps = [
                "Instalar dependencias (npm/pnpm/yarn según existan).",
                "Ejecutar tests/lint (si existen scripts).",
                "Validar build (si existe).",
                "Aplicar cambios y verificar nuevamente.",
            ]
            cmds = [
                "npm test",
                "npm run lint",
            ]
        elif ptype == "rust":
            steps = [
                "Compilar (cargo build) y ejecutar tests (cargo test).",
                "Revisar clippy si existe.",
                "Aplicar cambios y volver a compilar.",
            ]
            cmds = [
                "cargo test",
                "cargo clippy",
            ]
        else:
            steps = [
                "Determinar dependencias y comandos disponibles en el repo.",
                "Ejecutar tests/validaciones si existen.",
                "Aplicar cambios y re-verificar.",
            ]
            cmds = []

        payload = {
            "root": root_resolved,
            "project_type": ptype,
            "steps": steps,
            "commands": cmds if include_commands else [],
        }
        return json.dumps(payload, ensure_ascii=False)
    except Exception as e:
        return f"Error en project_workflow_suggest: {e}"


def fuzzy_search_paths(
    query: str,
    root: str = ".",
    exts: Optional[str] = None,
    max_results: int = 30,
    max_files: int = 5000,
) -> str:
    """
    Búsqueda "fuzzy" por nombres de archivos/rutas para encontrar candidatos rápidos.
    """
    try:
        q = (query or "").strip()
        if not q:
            return "Error: query vacía."
        resolved_root = resolve_path(root, must_exist=True)
        if resolved_root.startswith("Error:"):
            return resolved_root
        base = Path(resolved_root).resolve()

        allowed_exts: Optional[set[str]] = None
        if exts:
            allowed_exts = {e.strip().lower() for e in exts.split(",") if e.strip()}

        qn = _norm_text(q)
        scored: list[tuple[float, str]] = []
        seen = 0
        for p in base.rglob("*"):
            if seen >= max_files:
                break
            if p.is_dir():
                continue
            if allowed_exts is not None:
                if p.suffix.lower() not in allowed_exts:
                    continue
            seen += 1
            name_norm = _norm_text(p.name)
            score = difflib.SequenceMatcher(None, qn, name_norm).ratio()
            if score >= 0.25:
                scored.append((score, str(p)))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_results]
        lines = [f"{s:.3f} {path}" for s, path in top]
        if not lines:
            return "Sin coincidencias relevantes."
        return "\n".join(lines)
    except Exception as e:
        return f"Error en fuzzy_search_paths: {e}"


def build_text_index(
    root: str = ".",
    exts: Optional[str] = None,
    max_files: int = 400,
    max_chars_per_file: int = 20000,
    index_path: Optional[str] = None,
) -> str:
    """
    Construye un índice local de texto (RAG sin embeddings) con caché.
    """
    try:
        if _read_only_mode():
            return "Error: AARIS_READ_ONLY=true. build_text_index deshabilitado."

        resolved_root = resolve_path(root, must_exist=True)
        if resolved_root.startswith("Error:"):
            return resolved_root
        base = Path(resolved_root).resolve()

        if index_path is None:
            index_path = os.path.join(os.getcwd(), ".aaris", "rag_text_index.json")
        index_path = os.path.abspath(os.path.expanduser(index_path))
        Path(os.path.dirname(index_path)).mkdir(parents=True, exist_ok=True)

        allowed_exts: Optional[set[str]] = None
        if exts:
            allowed_exts = {e.strip().lower() for e in exts.split(",") if e.strip()}

        entries: list[dict[str, Any]] = []
        count = 0
        for p in base.rglob("*"):
            if count >= max_files:
                break
            if p.is_dir():
                continue
            if allowed_exts is not None and p.suffix.lower() not in allowed_exts:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if len(text) > max_chars_per_file:
                text = text[:max_chars_per_file] + "\n[…truncado…]"
            try:
                st = p.stat()
                mtime = st.st_mtime
                size = st.st_size
            except Exception:
                mtime = 0.0
                size = 0

            entries.append(
                {
                    "path": str(p),
                    "size": size,
                    "mtime": mtime,
                    "text_excerpt": text,
                }
            )
            count += 1

        payload = {
            "root": str(base),
            "created_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "exts": exts,
            "max_files": max_files,
            "max_chars_per_file": max_chars_per_file,
            "entries": entries,
        }
        Path(index_path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return f"OK build_text_index. entries={len(entries)} index={index_path}"
    except Exception as e:
        return f"Error en build_text_index: {e}"


def rag_query(
    query: str,
    index_path: Optional[str] = None,
    top_k: int = 6,
    max_excerpt_chars: int = 4000,
) -> str:
    """
    Consulta RAG local contra un índice de texto cacheado.
    """
    try:
        if not query or not query.strip():
            return "Error: query vacía."
        if index_path is None:
            index_path = os.path.join(os.getcwd(), ".aaris", "rag_text_index.json")
        index_path = os.path.abspath(os.path.expanduser(index_path))
        if not os.path.isfile(index_path):
            return "Error: no existe el índice. Ejecuta build_text_index primero."
        payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
        entries = payload.get("entries") or []
        qn = _norm_text(query)

        scored: list[tuple[float, dict[str, Any]]] = []
        for ent in entries:
            text_excerpt = ent.get("text_excerpt") or ""
            name_score = difflib.SequenceMatcher(None, qn, _norm_text(Path(ent.get("path") or "").name)).ratio()
            text_score = difflib.SequenceMatcher(None, qn, _norm_text(text_excerpt[:20000])).ratio()
            score = max(name_score, text_score)
            if score >= 0.15:
                scored.append((score, ent))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]
        if not top:
            return "Sin coincidencias en índice."
        out = []
        for score, ent in top:
            excerpt = ent.get("text_excerpt") or ""
            if len(excerpt) > max_excerpt_chars:
                excerpt = excerpt[:max_excerpt_chars] + "\n[…truncado…]"
            out.append({"score": score, "path": ent.get("path"), "excerpt": excerpt})
        return json.dumps({"top_k": top_k, "results": out}, ensure_ascii=False)
    except Exception as e:
        return f"Error en rag_query: {e}"


def apply_template(template_name: str, destination: str, context_json: str = "{}") -> str:
    """
    Crea un archivo desde una plantilla embebida.
    """
    try:
        if _read_only_mode():
            return "Error: AARIS_READ_ONLY=true. apply_template deshabilitado."
        ctx = json.loads(context_json or "{}")
        if not isinstance(ctx, dict):
            ctx = {}

        template_dir = os.environ.get(
            "AARIS_TEMPLATE_DIR",
            os.path.join(os.path.expanduser("~"), ".aaris", "templates"),
        )
        template_dir = os.path.abspath(os.path.expanduser(template_dir))

        templates: dict[str, str] = {
            "python_script": (
                "#!/usr/bin/env python3\n"
                "\"\"\"{{title}}\"\"\"\n\n"
                "def main():\n"
                "    print(\"{{message}}\")\n\n"
                "if __name__ == \"__main__\":\n"
                "    main()\n"
            ),
            "systemd_service": (
                "[Unit]\n"
                "Description={{description}}\n"
                "After=network.target\n\n"
                "[Service]\n"
                "Type=simple\n"
                "User={{user}}\n"
                "WorkingDirectory={{workdir}}\n"
                "ExecStart={{execstart}}\n"
                "Restart=always\n\n"
                "[Install]\n"
                "WantedBy=multi-user.target\n"
            ),
            "systemd_timer": (
                "[Unit]\n"
                "Description={{description}}\n\n"
                "[Timer]\n"
                "OnCalendar={{oncalendar}}\n"
                "Persistent=true\n\n"
                "[Install]\n"
                "WantedBy=timers.target\n"
            ),
            "bash_script": (
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n\n"
                "{{body}}\n"
            ),
            "cron_entry": (
                "{{minute}} {{hour}} {{day_of_month}} {{month}} {{day_of_week}} {{command}}\n"
            ),
            "readme": (
                "# {{title}}\n\n"
                "{{summary}}\n"
            ),
        }

        # Si existe plantilla externa, la usamos primero.
        external_content: Optional[str] = None
        for candidate in [template_name, f"{template_name}.tpl", f"{template_name}.txt"]:
            p = Path(template_dir) / candidate
            if p.is_file():
                try:
                    external_content = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    external_content = None
                break

        if external_content is not None:
            content = external_content
        else:
            if template_name not in templates:
                return f"Error: plantilla desconocida: {template_name}"
            content = templates[template_name]

        resolved_dest = resolve_path(destination, must_exist=False)
        if resolved_dest.startswith("Error:"):
            return resolved_dest
        for k, v in ctx.items():
            content = content.replace("{{" + str(k) + "}}", str(v))

        # Creamos directorios padre y escribimos.
        parent = os.path.dirname(resolved_dest)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(resolved_dest, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Plantilla aplicada en {resolved_dest} (template={template_name})."
    except Exception as e:
        return f"Error en apply_template: {e}"


def append_file(path: str, content: str) -> str:
    """
    Añade contenido al final de un archivo (crea directorios padre).
    """
    try:
        if _read_only_mode():
            return "Error: AARIS_READ_ONLY=true. append_file deshabilitado."
        resolved = resolve_path(path, must_exist=False)
        if resolved.startswith("Error:"):
            return resolved
        abs_path = resolved
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Para symlinks existentes, evitamos escapar del home.
        if os.path.isfile(abs_path):
            sym_err = _validate_symlink_for_path(abs_path, allow_escape=_allow_symlink_escape())
            if sym_err:
                return sym_err
        token = None
        if os.path.isfile(abs_path):
            token = _create_file_backup(Path(abs_path))
        with _path_lock(abs_path):
            with open(abs_path, "a", encoding="utf-8") as f:
                f.write(content)
        _read_file_cached.cache_clear()
        _exists_path_cached.cache_clear()
        if token:
            return f"Contenido añadido a {abs_path}. ROLLBACK_TOKEN={token}"
        return f"Contenido añadido a {abs_path}."
    except Exception as e:
        return f"Error en append_file: {e}"


def insert_after(path: str, anchor_text: str, insert_text: str, occurrence_index: int = 0) -> str:
    """
    Inserta `insert_text` justo después de la `occurrence_index`-ésima aparición de `anchor_text`.
    """
    try:
        if _read_only_mode():
            return "Error: AARIS_READ_ONLY=true. insert_after deshabilitado."
        resolved = resolve_path(path, must_exist=True)
        if resolved.startswith("Error:"):
            return resolved
        abs_path = resolved
        sym_err = _validate_symlink_for_path(abs_path, allow_escape=_allow_symlink_escape())
        if sym_err:
            return sym_err
        with _path_lock(abs_path):
            token = _create_file_backup(Path(abs_path))
            with open(abs_path, encoding="utf-8", errors="replace") as f:
                data = f.read()
            idx = -1
            start = 0
            for i in range(occurrence_index + 1):
                idx = data.find(anchor_text, start)
                if idx == -1:
                    return "Error: anchor_text no encontrado (o occurrence_index fuera de rango)."
                start = idx + len(anchor_text)
            insert_pos = idx + len(anchor_text)
            new_data = data[:insert_pos] + insert_text + data[insert_pos:]
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(new_data)
            if abs_path.lower().endswith(".py"):
                rb = validate_python_syntax(abs_path)
                if rb.startswith("Error"):
                    rollback_res = rollback(token, overwrite=True)
                    return f"Error: validación Python falló en {abs_path}. {rollback_res}"
            if abs_path.lower().endswith(".json"):
                try:
                    json.loads(new_data)
                except Exception:
                    rollback_res = rollback(token, overwrite=True)
                    return f"Error: JSON inválido en {abs_path}. {rollback_res}"
            _read_file_cached.cache_clear()
            _exists_path_cached.cache_clear()
            return f"Insertado después de anchor_text en {abs_path}. ROLLBACK_TOKEN={token}"
    except Exception as e:
        return f"Error en insert_after: {e}"


def _ensure_in_home_or_allow(target: Path, allow_dangerous: bool, action: str) -> Optional[str]:
    home = Path.home().resolve()
    if _is_relative_to(target, home):
        return None
    if allow_dangerous:
        return None
    return f"Error: {action} fuera de tu home requiere allow_dangerous=true."


def copy_path(from_path: str, to_path: str, overwrite: bool = False, allow_dangerous: bool = False) -> str:
    """
    Copia un archivo o carpeta a otra ubicación.
    """
    try:
        if _read_only_mode():
            return "Error: AARIS_READ_ONLY=true. copy_path deshabilitado."
        src = Path(resolve_path(from_path, must_exist=True))
        dst = Path(resolve_path(to_path, must_exist=False))
        if src.is_symlink():
            sym_err = _validate_symlink_for_path(str(src), allow_escape=_allow_symlink_escape())
            if sym_err:
                return sym_err
        err = _ensure_in_home_or_allow(src, allow_dangerous, "copy_path (origen)")
        if err:
            return err
        err = _ensure_in_home_or_allow(dst, allow_dangerous, "copy_path (destino)")
        if err:
            return err

        if dst.exists() and not overwrite:
            return f"Error: el destino ya existe: {dst}. Usa overwrite=true."

        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(str(src), str(dst), dirs_exist_ok=overwrite)
        else:
            shutil.copy2(str(src), str(dst))
        return f"Copia creada: {dst}"
    except Exception as e:
        return f"Error en copy_path: {e}"


def move_path(from_path: str, to_path: str, overwrite: bool = False, allow_dangerous: bool = False) -> str:
    """
    Mueve un archivo o carpeta a otra ubicación.
    """
    try:
        if _read_only_mode():
            return "Error: AARIS_READ_ONLY=true. move_path deshabilitado."
        src = Path(resolve_path(from_path, must_exist=True))
        dst = Path(resolve_path(to_path, must_exist=False))
        if src.is_symlink():
            sym_err = _validate_symlink_for_path(str(src), allow_escape=_allow_symlink_escape())
            if sym_err:
                return sym_err
        err = _ensure_in_home_or_allow(src, allow_dangerous, "move_path (origen)")
        if err:
            return err
        err = _ensure_in_home_or_allow(dst, allow_dangerous, "move_path (destino)")
        if err:
            return err

        if dst.exists():
            if not overwrite:
                return f"Error: el destino ya existe: {dst}. Usa overwrite=true."
            # Eliminamos destino para permitir el movimiento.
            if dst.is_dir():
                shutil.rmtree(str(dst))
            else:
                dst.unlink()

        token = _create_move_backup(src, dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        return f"Movido a: {dst}. ROLLBACK_TOKEN={token}"
    except Exception as e:
        return f"Error en move_path: {e}"


def _trash_enabled() -> bool:
    val = os.environ.get("AARIS_USE_TRASH", "true").strip().lower()
    return val not in ("0", "false", "no", "off")


def _trash_move(target: Path) -> str:
    """
    Mueve la ruta al directorio de papelera estándar (XDG Trash).
    Retorna un mensaje; si falla, lanza excepción para que el llamador haga fallback.
    """
    trash_base = Path.home().resolve() / ".local" / "share" / "Trash"
    files_dir = trash_base / "files"
    info_dir = trash_base / "info"
    files_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)

    original_path = str(target.resolve())
    uid = uuid.uuid4().hex
    dest_name = f"{target.name}.{uid}"
    dest = files_dir / dest_name

    shutil.move(str(target), str(dest))

    deletion_date = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
    # Formato .trashinfo (simplificado pero compatible).
    info_text = (
        "[Trash Info]\n"
        f"Path={original_path}\n"
        f"DeletionDate={deletion_date}\n"
    )
    info_file = info_dir / f"{dest_name}.trashinfo"
    info_file.write_text(info_text, encoding="utf-8")
    return str(dest)


def delete_path(
    path: str,
    recursive: bool = False,
    confirm: bool = False,
    allow_dangerous: bool = False,
    glob_filter: Optional[str] = None,
    max_entries: Optional[int] = None,
) -> str:
    """
    Borra un archivo o carpeta.

    - Para carpetas usa `recursive=true`.
    - Para carpetas (y para cosas fuera de tu home) exige `confirm=true` y/o `allow_dangerous=true`.
    """
    try:
        if _read_only_mode():
            return "Error: AARIS_READ_ONLY=true. delete_path deshabilitado."
        resolved = resolve_path(path, must_exist=True)
        if resolved.startswith("Error:"):
            return resolved

        target = Path(resolved).resolve()
        home = Path.home().resolve()
        pol_reason = _policy_forbidden_reason(str(target))
        if pol_reason and not allow_dangerous:
            return pol_reason
        if _policy_require_confirm("delete_path") and not confirm and not allow_dangerous:
            return "Error: política requiere confirm=true para delete_path."

        # Evita accidentes catastróficos.
        if str(target) == "/":
            return "Error: no se permite borrar `/`."

        min_bytes_for_confirm = int(os.environ.get("AARIS_DELETE_CONFIRM_MIN_BYTES", "50000000"))
        min_entries_for_confirm = int(os.environ.get("AARIS_DELETE_CONFIRM_MIN_ENTRIES", "200"))
        sensitive_dir_names = {".ssh", ".gnupg", ".kube", ".kubernetes"}
        sensitive_file_names = {".bashrc", ".zshrc", ".profile", ".bash_profile", ".gitconfig", ".config"}

        def _maybe_require_confirm(reason: str) -> Optional[str]:
            if confirm:
                return None
            return (
                "Confirmación requerida: " + reason + ". "
                "Repite la petición con `confirm=true`."
            )

        if target.is_dir():
            if not recursive:
                return f"Error: {target} es un directorio. Usa `recursive=true`."
            if not _is_relative_to(target, home) and not allow_dangerous:
                return "Error: borrar directorios fuera de tu home requiere `allow_dangerous=true`."

            # Confirmación estricta: cualquier borrado recursivo requiere confirm=true.
            if not confirm:
                return (
                    f"Confirmación requerida: vas a borrar recursivamente `{target}`. "
                    "Repite la petición con `confirm=true`."
                )

            # Selección segura para directorios grandes (si no se especifica filtro).
            effective_threshold = max_entries if max_entries is not None else int(
                os.environ.get("AARIS_DELETE_LARGE_DIR_MAX_ENTRIES", "2000")
            )
            if glob_filter is None and effective_threshold > 0:
                immediate = 0
                try:
                    for _ in target.iterdir():
                        immediate += 1
                        if immediate > effective_threshold:
                            break
                except Exception:
                    immediate = effective_threshold + 1
                if immediate > effective_threshold:
                    return (
                        "Error: el directorio parece demasiado grande para borrado recursivo directo. "
                        "Usa una subruta más específica o pasa `glob_filter` para borrar solo partes."
                    )

            # Si hay filtro, borramos recursivamente SOLO los hijos inmediatos que coincidan.
            if glob_filter:
                matches: list[Path] = []
                for child in target.iterdir():
                    if fnmatch.fnmatch(child.name, glob_filter):
                        matches.append(child)
                if not matches:
                    return f"Sin coincidencias para glob_filter={glob_filter!r} en {target}."

                if _trash_enabled():
                    moved = []
                    tokens: list[str] = []
                    for m in matches:
                        try:
                            original = m.resolve()
                            moved_dest = _trash_move(m)
                            moved.append(moved_dest)
                            tokens.append(_create_trash_backup(original, Path(moved_dest)))
                        except Exception:
                            # Fallback permanente por cada entrada.
                            if m.is_dir():
                                shutil.rmtree(str(m))
                            else:
                                m.unlink()
                    tok_part = f" ROLLBACK_TOKENS={','.join(tokens)}" if tokens else ""
                    return f"Entrada(s) movida(s) a Trash: {len(moved)}.{tok_part}"
                else:
                    for m in matches:
                        if m.is_dir():
                            shutil.rmtree(str(m))
                        else:
                            m.unlink()
                    return f"Entrada(s) eliminada(s): {len(matches)}"

            # Sin filtro: eliminamos todo (a Trash si está activo).
            if _trash_enabled():
                try:
                    original = target.resolve()
                    dest = _trash_move(target)
                    token = _create_trash_backup(original, Path(dest))
                    return f"Directorio movido a Trash: {dest}. ROLLBACK_TOKEN={token}"
                except Exception:
                    # Fallback permanente.
                    shutil.rmtree(str(target))
                    return f"Directorio eliminado: {target}"

            shutil.rmtree(str(target))
            return f"Directorio eliminado: {target}"

        # Archivo (o enlace).
        if not target.exists():
            return f"Error: la ruta no existe: {target}"

        if not _is_relative_to(target, home) and not allow_dangerous:
            return "Error: borrar archivos fuera de tu home requiere `allow_dangerous=true`."

        if not confirm:
            name = target.name
            parent = target.parent.name
            try:
                size = target.lstat().st_size
            except Exception:
                size = 0

            if parent in sensitive_dir_names or name in sensitive_file_names:
                msg = _maybe_require_confirm(f"es un archivo sensible (`{parent}/{name}`)")
                if msg:
                    return msg
            if name.startswith("."):
                msg = _maybe_require_confirm(f"es un archivo oculto (`{name}`)")
                if msg:
                    return msg
            if size >= min_bytes_for_confirm:
                msg = _maybe_require_confirm(f"es grande (>= {min_bytes_for_confirm} bytes)")
                if msg:
                    return msg

        if _trash_enabled():
            try:
                original = target.resolve()
                dest = _trash_move(target)
                token = _create_trash_backup(original, Path(dest))
                return f"Archivo movido a Trash: {dest}. ROLLBACK_TOKEN={token}"
            except Exception:
                target.unlink()
                return f"Archivo eliminado: {target}"

        target.unlink()
        return f"Archivo eliminado: {target}"

    except Exception as e:
        return f"Error en delete_path: {e}"


def run_command(
    command: str,
    cwd: Optional[str] = None,
    timeout_seconds: int = 120,
    max_output_chars: int = 24000,
    allow_dangerous: bool = False,
) -> str:
    """
    Ejecuta un comando de shell en Linux. Usar con cuidado: el usuario es responsable de lo que ejecuta.

    Args:
        command: Comando completo (como en una terminal).
        cwd: Directorio de trabajo; None = directorio actual del asistente.
        timeout_seconds: Tiempo máximo de ejecución.
        max_output_chars: Límite de salida combinada (stdout+stderr).
        allow_dangerous: Si False, bloquea heurísticamente comandos potencialmente destructivos.
    """
    try:
        if _read_only_mode():
            return "Error: AARIS_READ_ONLY=true. run_command deshabilitado."

        if not allow_dangerous:
            # Heurísticas para evitar ejecutar borrados/reescrituras masivas sin confirmación explícita.
            dangerous_patterns = [
                r"\brm\s+-rf\b",
                r"\brm\s+-r[f]?\b",
                r"\bdd\s+if=",
                r"\bmkfs\w*\b",
                r"\bshutdown\b",
                r"\breboot\b",
                r"\bpoweroff\b",
                r"\bhalt\b",
                r"\bkillall\b.*\b-9\b",
                r"\bkill\s+-9\b.*\b-1\b",
                r"\b:(){\s*:|\s*&};\s*:",
                r"\bxargs\b.*\brm\b",
            ]
            for pat in dangerous_patterns:
                if re.search(pat, command):
                    return (
                        "Error: comando potencialmente destructivo detectado. "
                        "Para ejecutarlo de todos modos, repite la petición haciendo que el agente llame a "
                        "`run_command(..., allow_dangerous=true)`."
                    )

        allowlist_only = os.environ.get("AARIS_COMMAND_ALLOWLIST_ONLY", "false").strip().lower() in (
            "1",
            "true",
            "yes",
            "si",
            "sí",
            "on",
        )
        if allowlist_only:
            allowlist_raw = os.environ.get(
                "AARIS_COMMAND_ALLOWLIST",
                "ls,cat,head,tail,stat,du,df,rg,systemctl,journalctl,ps,pwd,echo,whoami",
            )
            allowed = {x.strip() for x in allowlist_raw.split(",") if x.strip()}
            cmd = (command or "").strip()
            if cmd.startswith("sudo "):
                cmd = cmd[len("sudo ") :].lstrip()
            # Cuando usamos allowlist_only, bloqueamos redirecciones/tuberías/comandos compuestos.
            disallowed_chars = ["|", ";", "&&", "||", ">", "<", "\n", "\r", "`", "$(", "&"]
            for token in disallowed_chars:
                if token in cmd:
                    return "Error: AARIS_COMMAND_ALLOWLIST_ONLY no permite operadores de shell (tuberías/redirecciones/composición)."
            parts = cmd.split()
            first = parts[0] if parts else ""
            second = parts[1] if len(parts) > 1 else ""
            # Reglas simples para herramientas comunes.
            if first == "systemctl" and second not in ("status", "show"):
                return "Error: AARIS_COMMAND_ALLOWLIST_ONLY. systemctl restringido a status/show."
            if first == "journalctl" and second and not second.startswith("--since"):
                # No bloqueamos demasiado: pero restringimos el primer argumento.
                pass
            if first and first not in allowed:
                return f"Error: AARIS_COMMAND_ALLOWLIST_ONLY. comando no permitido: {first}"

        if cwd:
            resolved_cwd = resolve_path(cwd, must_exist=True)
            if resolved_cwd.startswith("Error:"):
                return resolved_cwd
            work = os.path.abspath(resolved_cwd)
        else:
            work = None
        if work and not os.path.isdir(work):
            return f"Error: cwd no es un directorio: {work}"
        sandbox_mode = os.environ.get("AARIS_COMMAND_SANDBOX", "").strip().lower()
        if sandbox_mode == "firejail":
            if not shutil.which("firejail"):
                return "Error: AARIS_COMMAND_SANDBOX=firejail pero `firejail` no está instalado."
            # Ejecutamos dentro de firejail sin red y con home privado.
            proc = subprocess.run(
                ["firejail", "--quiet", "--net=none", "--private-home", "sh", "-lc", command],
                shell=False,
                cwd=work,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env={**os.environ},
            )
        else:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=work,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env={**os.environ},
            )
        out = ""
        if proc.stdout:
            out += proc.stdout
        if proc.stderr:
            out += ("\n--- stderr ---\n" if out else "") + proc.stderr
        exit_note = f"\n[código de salida: {proc.returncode}]"
        if len(out) > max_output_chars:
            out = out[:max_output_chars] + f"\n[… salida truncada a {max_output_chars} caracteres]"
        return (out if out.strip() else "(sin salida)") + exit_note
    except subprocess.TimeoutExpired:
        return f"Error: el comando superó {timeout_seconds}s y fue cancelado."
    except Exception as e:
        return f"Error al ejecutar comando: {e}"


def run_command_checked(
    command: str,
    cwd: Optional[str] = None,
    timeout_seconds: int = 120,
    max_output_chars: int = 24000,
    allow_dangerous: bool = False,
) -> str:
    """
    Ejecuta comando y devuelve JSON con returncode/stdout/stderr (útil para loops/criterios).
    """
    try:
        if _read_only_mode():
            return "Error: AARIS_READ_ONLY=true. run_command_checked deshabilitado."

        # Reutilizamos run_command para respetar heurísticas y sandbox.
        out = run_command(
            command=command,
            cwd=cwd,
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
            allow_dangerous=allow_dangerous,
        )
        # run_command incluye el returncode al final en la forma:
        # \n[código de salida: N]
        m = re.search(r"\[código de salida:\s*(-?\d+)\]$", out.strip())
        returncode = int(m.group(1)) if m else None
        return json.dumps({"command": command, "returncode": returncode, "output": out}, ensure_ascii=False)
    except Exception as e:
        return f"Error en run_command_checked: {e}"


def run_command_retry(
    command: str,
    attempts: int = 3,
    delay_seconds: float = 1.0,
    cwd: Optional[str] = None,
    timeout_seconds: int = 120,
    max_output_chars: int = 12000,
    allow_dangerous: bool = False,
) -> str:
    """
    Ejecuta un comando varias veces hasta éxito (returncode==0) o agotar intentos.
    """
    try:
        if attempts < 1:
            return "Error: attempts debe ser >= 1."
        last = None
        for i in range(1, attempts + 1):
            res = run_command_checked(
                command=command,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_output_chars=max_output_chars,
                allow_dangerous=allow_dangerous,
            )
            last = res
            try:
                obj = json.loads(res)
                rc = obj.get("returncode")
                if rc == 0:
                    return json.dumps({"attempt": i, "result": obj}, ensure_ascii=False)
            except Exception:
                pass
            if i < attempts:
                import time

                time.sleep(delay_seconds)
        return json.dumps({"attempts": attempts, "last_result": json.loads(last) if last else None}, ensure_ascii=False)
    except Exception as e:
        return f"Error en run_command_retry: {e}"


def _sanitize_pkg_token(token: str) -> Optional[str]:
    t = (token or "").strip()
    if not t:
        return None
    # Paquetes típicamente usan [a-zA-Z0-9+._-]
    if not re.match(r"^[a-zA-Z0-9+._:-]+$", t):
        return None
    return t


def install_packages(
    packages: str,
    manager: str = "auto",
    update: bool = False,
    use_sudo: bool = False,
    assume_yes: bool = True,
    confirm: bool = False,
    allow_dangerous: bool = False,
) -> str:
    """
    Instala paquetes del sistema de forma segura (bloquea en modo read-only).

    - `packages`: lista separada por comas/espacios.
    - `confirm`: requerido cuando `update=true` o cuando el modo del sistema lo exija.
    - `allow_dangerous`: si true, permite operaciones más agresivas.
    """
    try:
        if _read_only_mode():
            return "Error: AARIS_READ_ONLY=true. install_packages deshabilitado."

        tokens = [t.strip() for t in (packages or "").replace(";", ",").replace("|", ",").split(",")]
        tokens = [t for t in tokens if t]
        if len(tokens) == 1 and " " in tokens[0]:
            tokens = [x.strip() for x in tokens[0].split() if x.strip()]

        pkg_tokens: list[str] = []
        for tok in tokens:
            tok_s = _sanitize_pkg_token(tok)
            if tok_s:
                pkg_tokens.append(tok_s)
        if not pkg_tokens:
            return "Error: no pude parsear tokens de paquetes válidos."

        if not confirm and (update or not allow_dangerous):
            return "Error: confirm=true requerido para install_packages (especialmente si update=true)."

        m = (manager or "auto").strip().lower()
        if m == "auto":
            if shutil.which("pacman"):
                m = "pacman"
            elif shutil.which("apt-get"):
                m = "apt"
            elif shutil.which("dnf"):
                m = "dnf"
            elif shutil.which("yum"):
                m = "yum"
            else:
                return "Error: no detecté un gestor de paquetes soportado (pacman/apt/dnf/yum)."

        sudo_prefix: list[str] = ["sudo"] if use_sudo else []
        if m == "pacman":
            cmds: list[list[str]] = []
            if update:
                cmds.append(sudo_prefix + ["pacman", "-Sy", "--noconfirm"])
            cmd_install = sudo_prefix + ["pacman", "-S", "--noconfirm"] + pkg_tokens
            cmds.append(cmd_install)
            outputs = []
            for c in cmds:
                proc = subprocess.run(c, capture_output=True, text=True, timeout=900, env={**os.environ})
                outputs.append(proc.stdout + (("\n--- stderr ---\n" + proc.stderr) if proc.stderr else ""))
                if proc.returncode != 0:
                    return "Error install_packages pacman:\n" + "\n".join(outputs)
            return "OK install_packages pacman.\n" + "\n".join(outputs)[:24000]

        if m == "apt":
            cmds = []
            if update:
                cmds.append(sudo_prefix + ["apt-get", "update", "-y"])
            # apt-get install -y requiere -y
            install_cmd = sudo_prefix + ["apt-get", "install"]
            if assume_yes:
                install_cmd.append("-y")
            install_cmd += pkg_tokens
            cmds.append(install_cmd)

            outputs = []
            for c in cmds:
                proc = subprocess.run(c, capture_output=True, text=True, timeout=900, env={**os.environ})
                outputs.append(proc.stdout + (("\n--- stderr ---\n" + proc.stderr) if proc.stderr else ""))
                if proc.returncode != 0:
                    return "Error install_packages apt:\n" + "\n".join(outputs)
            return "OK install_packages apt.\n" + "\n".join(outputs)[:24000]

        return f"Error: gestor no soportado: {m}"
    except Exception as e:
        return f"Error en install_packages: {e}"


def apply_unified_patch(
    path: str,
    patch_text: str,
    strip: int = 1,
    workdir: Optional[str] = None,
    confirm: bool = False,
    allow_dangerous: bool = False,
) -> str:
    """
    Aplica un diff unificado usando el binario `patch`.
    """
    try:
        if _read_only_mode():
            return "Error: AARIS_READ_ONLY=true. apply_unified_patch deshabilitado."
        if _policy_require_confirm("apply_unified_patch") and not confirm and not allow_dangerous:
            return "Error: política requiere confirm=true para apply_unified_patch."
        if not confirm and not allow_dangerous:
            return "Error: confirm=true requerido para apply_unified_patch."
        if not shutil.which("patch"):
            return "Error: no se encontró el binario `patch`."

        resolved_path = resolve_path(path, must_exist=True)
        if resolved_path.startswith("Error:"):
            return resolved_path
        target_file = Path(resolved_path)
        if not target_file.exists():
            return f"Error: no existe: {target_file}"
        token = _create_file_backup(target_file)
        pol_reason = _policy_forbidden_reason(str(target_file.resolve()))
        if pol_reason and not allow_dangerous:
            return pol_reason
        sym_err = _validate_symlink_for_path(str(target_file), allow_escape=_allow_symlink_escape())
        if sym_err:
            return sym_err

        if workdir is None:
            workdir = str(target_file.parent)
        resolved_workdir = resolve_path(workdir, must_exist=True)
        if resolved_workdir.startswith("Error:"):
            # si workdir no existe, usamos directorio padre del archivo
            resolved_workdir = str(target_file.parent.resolve())

        with _path_lock(str(target_file.resolve())):
            patch_tmp = Path(resolved_workdir) / f".aaris_patch_{uuid.uuid4().hex}.diff"
            patch_tmp.write_text(patch_text or "", encoding="utf-8")
            proc = subprocess.run(
                ["patch", f"-p{int(strip)}", "--batch", "-i", str(patch_tmp)],
                cwd=resolved_workdir,
                capture_output=True,
                text=True,
                timeout=300,
                env={**os.environ},
            )
            out = (proc.stdout or "") + (("\n--- stderr ---\n" + proc.stderr) if proc.stderr else "")
        try:
            if patch_tmp.exists():
                patch_tmp.unlink()
        except Exception:
            pass
        if proc.returncode != 0:
            return "Error apply_unified_patch:\n" + out[:12000]
        # Post-checks básicas.
        if str(target_file).lower().endswith(".json"):
            try:
                _ = json.loads(target_file.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                rollback_res = rollback(token, overwrite=True)
                return f"Error: apply_unified_patch dejó JSON inválido. {rollback_res}"
        if str(target_file).lower().endswith(".py"):
            rb = validate_python_syntax(str(target_file))
            if rb.startswith("Error"):
                rollback_res = rollback(token, overwrite=True)
                return f"Error: validación Python falló tras apply_unified_patch. {rollback_res}"
        return "OK apply_unified_patch.\n" + out[:12000] + f"\nROLLBACK_TOKEN={token}"
    except Exception as e:
        return f"Error en apply_unified_patch: {e}"

def ast_list_functions(path: str) -> str:
    """
    Lista las clases y funciones de un archivo Python usando parseo AST nativo.
    """
    try:
        resolved = resolve_path(path, must_exist=True)
        if resolved.startswith("Error:"): return resolved
        with open(resolved, "r", encoding="utf-8") as f:
            code = f.read()
        import ast
        tree = ast.parse(code)
        lines = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                lines.append(f"Function: {node.name}() (line {node.lineno})")
            elif isinstance(node, ast.AsyncFunctionDef):
                lines.append(f"AsyncFunction: {node.name}() (line {node.lineno})")
            elif isinstance(node, ast.ClassDef):
                lines.append(f"Class: {node.name} (line {node.lineno})")
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef):
                        lines.append(f"  Method: {sub.name}() (line {sub.lineno})")
        return "\n".join(lines) if lines else "No se encontraron funciones o clases en el nivel superior."
    except Exception as e:
        return f"Error en ast_list_functions: {e}"

def ast_read_function(path: str, func_name: str) -> str:
    """
    Extrae el código fuente exacto de una función o clase de un archivo Python usando AST.
    """
    try:
        resolved = resolve_path(path, must_exist=True)
        if resolved.startswith("Error:"): return resolved
        with open(resolved, "r", encoding="utf-8") as f:
            code = f.read()
        import ast
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == func_name:
                    if hasattr(ast, "unparse"): 
                        return ast.unparse(node)
                    return "Error: ast.unparse requiere Python 3.9+"
        return f"No se encontró {func_name} en {path}."
    except Exception as e:
        return f"Error en ast_read_function: {e}"

def docker_ps() -> str:
    """Ejecuta docker ps y devuelve el listado de contenedores corriendo."""
    try:
        proc = subprocess.run(["docker", "ps", "--format", "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}"], capture_output=True, text=True, timeout=10)
        if proc.returncode != 0: return f"Error: {proc.stderr}"
        return proc.stdout or "No containers running."
    except Exception as e:
        return f"Error en docker_ps (¿Docker está instalado y activo?): {e}"

def docker_logs(container: str, tail: int = 50) -> str:
    """Obtiene las últimas líneas de logs de un contenedor docker."""
    try:
        proc = subprocess.run(["docker", "logs", "--tail", str(tail), container], capture_output=True, text=True, timeout=15)
        return (proc.stdout + proc.stderr) or "(sin logs)"
    except Exception as e:
        return f"Error en docker_logs: {e}"

def docker_exec(container: str, command: str, allow_dangerous: bool = False) -> str:
    """Ejecuta un comando en un contenedor docker en ejecución."""
    try:
        if _policy_require_confirm("docker_exec") and not allow_dangerous:
            return "Error: confirmación o allow_dangerous=true requerido."
        proc = subprocess.run(["docker", "exec", container, "sh", "-c", command], capture_output=True, text=True, timeout=60)
        return (proc.stdout + proc.stderr) or f"Comando ejecutado con código {proc.returncode}."
    except Exception as e:
        return f"Error en docker_exec: {e}"

def db_query_sqlite(db_path: str, query: str) -> str:
    """Ejecuta una consulta SQL en una base de datos SQLite (.db, .sqlite)."""
    try:
        import sqlite3
        resolved = resolve_path(db_path, must_exist=True)
        if resolved.startswith("Error:"): return resolved
        if _read_only_mode() and not query.strip().upper().startswith("SELECT"):
            return "Error: AARIS_READ_ONLY=true, solo SELECT está permitido."
            
        with sqlite3.connect(resolved) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)
            if query.strip().upper().startswith("SELECT"):
                rows = cursor.fetchall()
                if not rows: return "0 results."
                res = [dict(r) for r in rows]
                return json.dumps(res, ensure_ascii=False)[:24000]
            else:
                conn.commit()
                return f"Affected rows: {cursor.rowcount}"
    except Exception as e:
        return f"Error en db_query_sqlite: {e}"

def delegate_task(prompt: str) -> str:
    """Delega una tarea a una nueva instancia del agente ejecutando main.py --run-prompt."""
    try:
        agent_script = str(Path(__file__).resolve().parent / "main.py")
        import sys
        proc = subprocess.run(
            [sys.executable, agent_script, "--run-prompt", prompt],
            capture_output=True, text=True, timeout=300, env={**os.environ}
        )
        out = proc.stdout + proc.stderr
        return f"Delegado ejecutado (código {proc.returncode}). Salida:\n{out[-12000:]}"
    except Exception as e:
        return f"Error en delegate_task: {e}"
        
def schedule_agent_task(cron_expr: str, prompt: str, task_name: str) -> str:
    """Crea una tarea programada usando crontab del usuario para lanzar el agente."""
    try:
        agent_script = str(Path(__file__).resolve().parent / "main.py")
        import shlex, sys
        prompt_q = shlex.quote(prompt)
        command = f"{sys.executable} {agent_script} --run-prompt {prompt_q} >> ~/.aaris/cron.log 2>&1"
        cron_line = f"{cron_expr} {command}\n"
        
        cron_file = Path.home() / ".aaris" / "crontab.txt"
        cron_file.parent.mkdir(parents=True, exist_ok=True)
        
        proc = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        current_cron = proc.stdout if proc.returncode == 0 else ""
        
        if cron_line in current_cron:
            return f"Tarea '{task_name}' ya estaba programada."
            
        new_cron = current_cron.strip() + "\n" + f"# AARIS_TASK: {task_name}\n{cron_line}"
        cron_file.write_text(new_cron, encoding="utf-8")
        
        apply_proc = subprocess.run(["crontab", str(cron_file)], capture_output=True, text=True)
        if apply_proc.returncode != 0:
            return f"Error aplicando crontab (fallback cron no disponible): {apply_proc.stderr}"
        return f"Tarea '{task_name}' programada exitosamente con cron '{cron_expr}'."
    except Exception as e:
        return f"Error en schedule_agent_task: {e}"
