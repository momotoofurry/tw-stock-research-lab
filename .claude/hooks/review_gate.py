"""提交閘門：沒有 diff-reviewer 子代理的 APPROVE，就不允許 git commit。

用法：
  python .claude/hooks/review_gate.py hash            印出目前暫存差異的 SHA256（審查員用）
  python .claude/hooks/review_gate.py approve <sha>   審查員回 APPROVE 後，把核准標記寫進 .git/（雜湊必須和目前暫存差異一致）
  python .claude/hooks/review_gate.py precommit       git pre-commit hook（.git/hooks/pre-commit 呼叫）：實際要提交的內容沒核准就擋
  python .claude/hooks/review_gate.py check           Claude 的 PreToolUse hook：擋下跳過 git hook 的寫法（--no-verify、hooksPath）

真正的檢查在 git 的 pre-commit：git 在寫入 commit 的那一刻執行它，這時的暫存區（含 commit -a、接檔案路徑、
同一行先 git add 再 commit 的情況）就是實際要提交的內容，所以指令怎麼寫都繞不過。
核准標記放在 .git/claude-review-approved（不會被提交）。內容只要有任何改動，雜湊就對不上，必須重審。
使用者自己在終端機手動 commit 也會被擋；真的要跳過時用 git commit --no-verify（Claude 不准用）。
限制：標記由主會話寫入、.git/hooks 也可以被改，這是防止「忘了審查就提交」的閘門，不是防範惡意繞過的安全機制。
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MARKER = ROOT / ".git" / "claude-review-approved"

# git ... commit（允許 git -c a=b commit、git -C dir commit、git.exe、大小寫不同）；不攔 git commit-tree 之類
COMMIT_RE = re.compile(r"\bgit(?:\.exe)?(?:\s+-[cC]\s+\S+|\s+--?[\w-]+(?:=\S+)?)*\s+commit(?![\w-])", re.IGNORECASE)
# 會帶下一個參數當值的 commit 選項
VALUE_OPTS = {"-m", "-F", "-C", "-c", "-t", "--message", "--file", "--reuse-message", "--reedit-message",
              "--author", "--date", "--template", "--cleanup", "--fixup", "--squash", "--trailer"}
SHORT_WITH_VALUE = "mFCct"   # -mMSG 這類黏在一起的短選項，遇到這些字母後面就是值
VALUE_LONG = [o[2:] for o in VALUE_OPTS if o.startswith("--")]
# 會跳過 git hook 或改動 hook 設定的寫法
HOOK_BYPASS_RE = re.compile(r"hooks?path|\.git[\\/]+hooks|claude-review-approved", re.IGNORECASE)


def commit_args(rest: str) -> list[str]:
    """commit 之後、下一個指令之前的參數；引號不成對時 shlex 會丟 ValueError。"""
    seps = ";&|\n"   # 引號外的換行也是指令分隔（PowerShell 常用）
    lex = shlex.shlex(rest, posix=True, punctuation_chars=seps)
    lex.whitespace = " \t\r"
    lex.whitespace_split = True
    toks = []
    for t in lex:   # 引號內的換行（heredoc 訊息、PowerShell here-string）留在同一個參數裡
        if set(t) <= set(seps):
            break
        toks.append(t)
    return toks


def skips_hooks(toks: list[str]) -> bool:
    """參數裡有沒有 --no-verify（含縮寫 --no-v）或短選項 -n（含 -nm 這類合寫）。"""
    skip = False
    for t in toks:
        if skip:
            skip = False
            continue
        if t == "--":
            break
        if t.startswith("--"):
            name = t[2:].split("=", 1)[0]
            if len(name) > 2 and "no-verify".startswith(name):
                return True
            if "=" not in t and any(v.startswith(name) for v in VALUE_LONG):
                skip = True   # 值在下一個參數（--message "x"、--mess "x"）
            continue
        if t in VALUE_OPTS:
            skip = True
            continue
        if t.startswith("-") and len(t) > 1:
            for i, ch in enumerate(t[1:]):
                if ch == "n":
                    return True
                if ch in SHORT_WITH_VALUE:
                    skip = i == len(t) - 2   # 值在下一個參數（如 -m "x"）；黏著的（-mx）就不用跳
                    break
    return False


def staged_hash() -> tuple[str, int]:
    """暫存差異的雜湊。在 pre-commit 裡 git 會把 GIT_INDEX_FILE 指向實際要提交的暫存區，這裡沿用環境變數。"""
    diff = subprocess.run(["git", "diff", "--cached", "--binary"], cwd=ROOT, capture_output=True, check=True).stdout
    return hashlib.sha256(diff).hexdigest(), len(diff)


def approved() -> str:
    return MARKER.read_text(encoding="utf-8").strip() if MARKER.exists() else ""


def precommit() -> None:
    try:
        current, size = staged_hash()
    except subprocess.CalledProcessError as e:
        print(f"[review_gate] 無法讀取要提交的內容（{e}），為安全起見不允許提交。", file=sys.stderr)
        sys.exit(1)
    if size == 0:
        return   # 沒有內容變動（例如只改 commit 訊息），沒有東西要審
    if approved() != current:
        print("[review_gate] 已攔下提交：要提交的內容沒有 diff-reviewer 的 APPROVE（或核准後又改過）。\n"
              "  1. git add 要提交的檔案（不要用 commit -a 或在 commit 後面接檔案路徑）\n"
              "  2. 呼叫 diff-reviewer 子代理審查\n"
              "  3. APPROVE 後執行 python .claude/hooks/review_gate.py approve <DIFF_SHA256>\n"
              f"  目前要提交的內容雜湊 {current}\n"
              "  （使用者本人真的要跳過時可用 git commit --no-verify）", file=sys.stderr)
        sys.exit(1)


def check() -> None:
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return
    cmd = (data.get("tool_input") or {}).get("command") or ""
    if HOOK_BYPASS_RE.search(cmd):
        print("[review_gate] 已攔下：不准修改 git hook 設定、.git/hooks 或核准標記。", file=sys.stderr)
        sys.exit(2)
    for m in COMMIT_RE.finditer(cmd):
        try:
            bad = skips_hooks(commit_args(cmd[m.end():]))
        except ValueError:
            bad = True   # 引號不成對之類，無法判斷就擋
        if bad:
            print("[review_gate] 已攔下提交：Claude 不准用 --no-verify / -n 跳過 pre-commit 審查閘門。", file=sys.stderr)
            sys.exit(2)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    op = sys.argv[1] if len(sys.argv) > 1 else "check"
    if op == "hash":
        print(staged_hash()[0])
    elif op == "approve":
        if len(sys.argv) < 3:
            sys.exit("用法：review_gate.py approve <DIFF_SHA256>")
        current = staged_hash()[0]
        if sys.argv[2].strip().lower() != current:
            sys.exit(f"雜湊不符：審查的是 {sys.argv[2]}，目前暫存差異是 {current}；請重新審查。")
        MARKER.write_text(current, encoding="utf-8")
        print(f"已核准暫存差異 {current}")
    elif op == "precommit":
        precommit()
    else:
        check()


if __name__ == "__main__":
    main()
