#!/usr/bin/env bash
# download the account-only CameraHMR files; safe to re-run

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

SMPL=data/models/SMPL/SMPL_NEUTRAL.pkl
FOV=data/camerahmr/cam_model_cleaned.ckpt

if [ -s "$SMPL" ] && [ -s "$FOV" ]; then
    echo "[skip] $SMPL (exists)"
    echo "[skip] $FOV (exists)"
    echo "All CameraHMR files in place."
    exit 0
fi

echo "Log in with your CameraHMR account (register at https://camerahmr.is.tue.mpg.de/)."
read -r -p "Username (CameraHMR): " username
read -r -s -p "Password (CameraHMR): " password; echo
username=$(urle "$username")
password=$(urle "$password")
# credentials via a private temp file
POSTFILE=$(umask 077; mktemp)
trap 'rm -f "$POSTFILE"' EXIT
printf 'username=%s&password=%s' "$username" "$password" > "$POSTFILE"
FAILED=()

get() {
    local dest="$1" url="$2"
    if [ -s "$dest" ]; then echo "[skip] $dest (exists)"; return 0; fi
    mkdir -p "$(dirname "$dest")"
    echo "[get ] $dest"
    # a failed login returns an HTML page
    if wget -q --show-progress --post-file "$POSTFILE" "$url" -O "$dest.part" --continue \
       && ! head -c 1024 "$dest.part" | grep -a -q -i -e '<html' -e '<!doctype' \
       && mv "$dest.part" "$dest"; then
        return 0
    fi
    rm -f "$dest.part"; FAILED+=("$dest"); return 1
}

get "$SMPL" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=SMPL_NEUTRAL.pkl'
get "$FOV" 'https://download.is.tue.mpg.de/download.php?domain=camerahmr&sfile=cam_model_cleaned.ckpt'

echo
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "DONE WITH ERRORS -- failed:"
    printf '  %s\n' "${FAILED[@]}"
    echo "Check the username/password and that the license is accepted on camerahmr.is.tue.mpg.de."
    exit 1
fi
echo "CameraHMR files in place."
echo "SMPL_NEUTRAL.pkl contains chumpy objects: convert it to plain NumPy once"
echo "(README, step 3)."
