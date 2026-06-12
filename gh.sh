# -------------------------------------------------------
# gh: Commit to dual GitHub repo
# Usage: gh <COMMIT COMMENT>
# On Linux, edit .bashrc and copy this entire file at the end
#	sudo nano ~/.bashrc	
# In nano, press CTR + X to exit (select Y on save)
# For mac computers, use .zshrc instead of .bashrc
# -------------------------------------------------------
dgh() {
    local VERSIONFILE="VERSION.txt"
    local WORKHASH
    local commit_msg

    # ================================================================
    # PULL-ONLY MODE  (no arguments)
    # ================================================================
    if [ "$#" -eq 0 ]; then
        echo "[dgh] No commit message supplied -- pulling only."
        git pull origin main || return 1

        if [ -f "$VERSIONFILE" ]; then
            WORKHASH=$(awk '{print $1}' "$VERSIONFILE")
            echo "[gh] Stamped commit in files: $WORKHASH"
        else
            echo "[gh] WARNING: $VERSIONFILE not found."
            WORKHASH="$(git rev-parse --short HEAD 2>/dev/null)" || return 1
        fi
        echo
        echo "================================================================="
        echo "         CONFIGURATION MANAGEMENT ACROSS BOUNDARIES"
        echo "================================================================="
        echo
        echo "  Before evaluating any inputs or producing outputs, search"
        echo "  project knowledge for \"GITCOMMIT $WORKHASH\" and confirm"
        echo "  that the search returns a result containing that exact string."
        echo
        echo "  Current commit: $WORKHASH"
        echo "================================================================="
        echo
        echo "  If the search returns an older hash or nothing, the project"
        echo "  knowledge base has not yet re-indexed -- wait ~5 minutes and"
        echo "  run gh (pull-only) to get a fresh prompt before proceeding."
        echo
        return 0
    fi

    # ================================================================
    # COMMIT MODE
    # ================================================================

    # Join all arguments into one commit message
    commit_msg="$*"

    # ---- Stage all changes ----
    git add -A || return 1

    # ---- Pull before committing to avoid divergence ----
    git pull origin main || return 1

    # ---- Commit only if there is something staged ----
    if ! git diff --cached --quiet; then
        git commit -m "$commit_msg" || return 1
    else
        echo "[gh] Nothing to commit -- working tree clean."
    fi

    # ---- Push to organization repo ----
    git push origin HEAD || return 1

    # ---- Push to personal repo ----
    git push backup HEAD || return 1

    # ---- Capture the WORK hash (this is what goes into the files) ----
    WORKHASH=$(git rev-parse --short HEAD) || return 1
    echo "[gh] Work commit: $WORKHASH"

    # ================================================================
    # VERSION STAMP
    # ================================================================

    # ---- Write VERSION ----
    printf '%s %s\n' "$WORKHASH" "$(date '+%Y-%m-%d %H:%M:%S %Z')" > "$VERSIONFILE"

    # ---- Stage and commit stamp file ----
    git add "$VERSIONFILE" || return 1

    if ! git diff --cached --quiet; then
        git commit -m "Auto: version stamp $WORKHASH" || return 1

        # ---- Push stamp commit to both remotes ----
        git push origin HEAD || return 1
        git push backup HEAD || return 1

        echo "[gh] Stamp commit pushed."
    fi

    # ---- Print the agent session prompt (always uses WORKHASH) ----
    echo
    echo "================================================================="
    echo "         CONFIGURATION MANAGEMENT ACROSS BOUNDARIES"
    echo "================================================================="
    echo
    echo "  Before evaluating any inputs or producing outputs, search"
    echo "  project knowledge for \"GITCOMMIT $WORKHASH\" and confirm"
    echo "  that the search returns a result containing that exact string."
    echo
    echo "  Current commit: $WORKHASH"
    echo "================================================================="
    echo
    echo "  If the search returns an older hash or nothing, the project"
    echo "  knowledge base has not yet re-indexed -- wait ~5 minutes and"
    echo "  run gh (pull-only) to get a fresh prompt before proceeding."
    echo
}