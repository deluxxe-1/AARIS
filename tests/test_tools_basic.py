import os
import re
import tempfile
import unittest
from pathlib import Path


# Import from workspace (assumes tests run from repo root or PYTHONPATH includes it).
from tools import delete_path, edit_file, move_path, rollback, project_workflow_suggest, policy_show


class TestToolsBasic(unittest.TestCase):
    def setUp(self) -> None:
        # Evitamos tocar la papelera real durante tests.
        self._old_trash = os.environ.get("AARIS_USE_TRASH")
        os.environ["AARIS_USE_TRASH"] = "false"

        self._tmp_backup = tempfile.mkdtemp(prefix="aaris_test_backups_")
        self._old_backup_path = os.environ.get("AARIS_BACKUP_PATH")
        os.environ["AARIS_BACKUP_PATH"] = self._tmp_backup

    def tearDown(self) -> None:
        import tools
        tools._POLICY_CACHE = None
        if self._old_trash is None:
            os.environ.pop("AARIS_USE_TRASH", None)
        else:
            os.environ["AARIS_USE_TRASH"] = self._old_trash

        if getattr(self, "_old_backup_path", None) is None:
            os.environ.pop("AARIS_BACKUP_PATH", None)
        else:
            os.environ["AARIS_BACKUP_PATH"] = self._old_backup_path

        try:
            # No importa si falla, el sandbox limpia al final.
            for p in Path(self._tmp_backup).rglob("*"):
                if p.is_file():
                    p.unlink()
            for p in sorted(Path(self._tmp_backup).rglob("*"), reverse=True):
                if p.is_dir():
                    p.rmdir()
            Path(self._tmp_backup).rmdir()
        except Exception:
            pass

    def test_edit_file_creates_rollback_token_and_restores(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.txt"
            p.write_text("hello\n", encoding="utf-8")

            res = edit_file(str(p), "hello world\n")
            m = re.search(r"ROLLBACK_TOKEN=([a-fA-F0-9]+)", res)
            self.assertIsNotNone(m, msg=res)
            token = m.group(1)

            # Confirmamos cambio.
            self.assertEqual(p.read_text(encoding="utf-8"), "hello world\n")

            # Rollback.
            rb = rollback(token, overwrite=True)
            self.assertIn("Rollback OK", rb)
            self.assertEqual(p.read_text(encoding="utf-8"), "hello\n")

    def test_delete_recursive_requires_confirm(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "dir"
            target.mkdir()
            (target / "x.txt").write_text("x", encoding="utf-8")

            res = delete_path(str(target), recursive=True, confirm=False, allow_dangerous=True)
            self.assertIn("Confirmación requerida", res)

    def test_move_path_creates_rollback_token_and_restores(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "src.txt"
            dst = Path(d) / "dst.txt"
            src.write_text("abc", encoding="utf-8")

            res = move_path(str(src), str(dst), overwrite=False, allow_dangerous=True)
            m = re.search(r"ROLLBACK_TOKEN=([a-fA-F0-9]+)", res)
            self.assertIsNotNone(m, msg=res)
            token = m.group(1)

            self.assertFalse(src.exists())
            self.assertTrue(dst.exists())

            rb = rollback(token, overwrite=False)
            self.assertIn("Rollback OK", rb)
            self.assertTrue(src.exists())
            self.assertFalse(dst.exists())
            self.assertEqual(src.read_text(encoding="utf-8"), "abc")

    def test_policy_show_returns_json(self):
        res = policy_show()
        self.assertTrue(res.strip().startswith("{"), msg=res)

    def test_project_workflow_suggest_returns_json(self):
        res = project_workflow_suggest(root=".", include_commands=False)
        self.assertTrue(res.strip().startswith("{"), msg=res)


if __name__ == "__main__":
    unittest.main()

