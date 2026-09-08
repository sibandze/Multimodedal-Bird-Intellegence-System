#!/usr/bin/env bash
# filemap - prints a full tree-like map from current directory

# usage:./filemap.sh [path] [--all]
# example:./filemap.sh
# example:./filemap.sh src
# example:./filemap.sh. --all

ROOT="."
SHOW_ALL=0

# Parse args
for arg in "$@"; do
  if [[ "$arg" == "--all" ]]; then
    SHOW_ALL=1
  else
    ROOT="$arg"
  fi
done

# Files/folders to ignore in the tree output
IGNORE_PATTERNS=(
  ".vscode"
  ".venv"
  "*cache"
  ".git"
  "__pycache__"
  "*.pyc"
  "*_review.md" # your review notes
  "*.egg-info"
  ".DS_Store"
  "node_modules"
  "*.ogg"
  "*.npy"
)

print_map() {
  local dir="$1" prefix="$2"

  local entries=()
  shopt -s nullglob dotglob
  mapfile -t entries < <(printf "%s\n" "$dir"/* "$dir"/.* 2>/dev/null | sort -u | grep -v -E "/\.$|/\.\.$")

  local filtered=()
  # Filter out ignored patterns unless --all
  for entry in "${entries[@]}"; do
    local name=$(basename "$entry")
    local skip=0
    if (( SHOW_ALL == 0 )); then
      for pat in "${IGNORE_PATTERNS[@]}"; do
        [[ "$name" == $pat ]] && skip=1 && break
      done
    fi
    (( skip == 0 )) && filtered+=("$entry")
  done

  local total=${#filtered[@]}
  local i=0
  for entry in "${filtered[@]}"; do
    ((i++))
    local name=$(basename "$entry")
    local connector="├──"
    local next_prefix="│ "
    (( i == total )) && { connector="└──"; next_prefix=" "; }

    if [ -d "$entry" ] && [ ! -L "$entry" ]; then
      echo "${prefix}${connector} ${name}/"
      print_map "$entry" "${prefix}${next_prefix}"
    else
      echo "${prefix}${connector} ${name}"
    fi
  done
}

if (( SHOW_ALL == 1 )); then
  echo "${ROOT}/ [ALL FILES]"
else
  echo "${ROOT}/"
fi

print_map "$ROOT" ""
