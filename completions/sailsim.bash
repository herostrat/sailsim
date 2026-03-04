# Bash completion for sailsim
#
# Install:
#   source completions/sailsim.bash
#
# Or copy to /etc/bash_completion.d/ (or ~/.local/share/bash-completion/completions/)

_sailsim_find_configs_root() {
    # Walk up from cwd to find configs/ directory
    local dir="$PWD"
    while [[ "$dir" != "/" ]]; do
        if [[ -d "$dir/configs" ]]; then
            echo "$dir/configs"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

_sailsim_profiles() {
    local subdir="$1"
    local configs_root
    configs_root="$(_sailsim_find_configs_root)" || return
    local dir="$configs_root/$subdir"
    if [[ -d "$dir" ]]; then
        local f
        for f in "$dir"/*.toml; do
            [[ -e "$f" ]] && basename "$f" .toml
        done
    fi
}

_sailsim() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    opts="--scenario --yacht --autopilot --output --save-json --view --quiet --help"

    case "$prev" in
        --scenario)
            # Complete with scenario TOML files
            local configs_root
            configs_root="$(_sailsim_find_configs_root)"
            if [[ -n "$configs_root" && -d "$configs_root/scenarios" ]]; then
                local files
                files=$(cd "$configs_root/scenarios" && compgen -f -X '!*.toml' -- "$cur")
                COMPREPLY=( $(printf "%s\n" $files | sed "s|^|$configs_root/scenarios/|") )
            fi
            COMPREPLY+=( $(compgen -f -X '!*.toml' -- "$cur") )
            compopt -o filenames 2>/dev/null
            return
            ;;
        --yacht)
            # Yacht profile names + .toml file completion
            local profiles
            profiles="$(_sailsim_profiles yachts)"
            COMPREPLY=( $(compgen -W "$profiles" -- "$cur") )
            COMPREPLY+=( $(compgen -f -X '!*.toml' -- "$cur") )
            compopt -o filenames 2>/dev/null
            return
            ;;
        --autopilot)
            # Autopilot profile names + .toml file completion
            local profiles
            profiles="$(_sailsim_profiles autopilots)"
            COMPREPLY=( $(compgen -W "$profiles" -- "$cur") )
            COMPREPLY+=( $(compgen -f -X '!*.toml' -- "$cur") )
            compopt -o filenames 2>/dev/null
            return
            ;;
        --output|--save-json)
            COMPREPLY=( $(compgen -f -- "$cur") )
            compopt -o filenames 2>/dev/null
            return
            ;;
        --view)
            COMPREPLY=( $(compgen -f -X '!*.json' -- "$cur") )
            compopt -o filenames 2>/dev/null
            return
            ;;
    esac

    # Default: complete flags
    if [[ "$cur" == -* ]]; then
        COMPREPLY=( $(compgen -W "$opts" -- "$cur") )
        return
    fi
}

complete -F _sailsim sailsim
complete -F _sailsim "python -m sailsim"

# --- Completion for analyze_autopilot.py ---

_analyze_autopilot() {
    local cur prev subcmd
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Find the subcommand
    subcmd=""
    for word in "${COMP_WORDS[@]}"; do
        case "$word" in
            analytical|empirical|full) subcmd="$word"; break ;;
        esac
    done

    # Complete subcommands
    if [[ -z "$subcmd" ]]; then
        COMPREPLY=( $(compgen -W "analytical empirical full --help" -- "$cur") )
        return
    fi

    case "$subcmd" in
        analytical)
            case "$prev" in
                --yacht)
                    local profiles
                    profiles="$(_sailsim_profiles yachts)"
                    COMPREPLY=( $(compgen -W "$profiles" -- "$cur") )
                    COMPREPLY+=( $(compgen -f -X '!*.toml' -- "$cur") )
                    compopt -o filenames 2>/dev/null
                    return ;;
                --autopilot)
                    local profiles
                    profiles="$(_sailsim_profiles autopilots)"
                    COMPREPLY=( $(compgen -W "$profiles" -- "$cur") )
                    COMPREPLY+=( $(compgen -f -X '!*.toml' -- "$cur") )
                    compopt -o filenames 2>/dev/null
                    return ;;
                --save-dir)
                    COMPREPLY=( $(compgen -d -- "$cur") )
                    compopt -o filenames 2>/dev/null
                    return ;;
                --speed|--omega-n|--zeta) return ;;
            esac
            if [[ "$cur" == -* ]]; then
                COMPREPLY=( $(compgen -W "--yacht --autopilot --speed --omega-n --zeta --sweep --save-dir --no-plot --help" -- "$cur") )
            fi
            ;;
        empirical)
            case "$prev" in
                --save-dir)
                    COMPREPLY=( $(compgen -d -- "$cur") )
                    compopt -o filenames 2>/dev/null
                    return ;;
                --rate-limit) return ;;
            esac
            if [[ "$cur" == -* ]]; then
                COMPREPLY=( $(compgen -W "--rate-limit --save-dir --no-plot --help" -- "$cur") )
            else
                COMPREPLY=( $(compgen -f -X '!*.json' -- "$cur") )
                compopt -o filenames 2>/dev/null
            fi
            ;;
        full)
            case "$prev" in
                --scenario)
                    local configs_root
                    configs_root="$(_sailsim_find_configs_root)"
                    if [[ -n "$configs_root" && -d "$configs_root/scenarios" ]]; then
                        local profiles
                        profiles=$(_sailsim_profiles scenarios)
                        COMPREPLY=( $(compgen -W "$profiles" -- "$cur") )
                    fi
                    COMPREPLY+=( $(compgen -f -X '!*.toml' -- "$cur") )
                    compopt -o filenames 2>/dev/null
                    return ;;
                --save-dir)
                    COMPREPLY=( $(compgen -d -- "$cur") )
                    compopt -o filenames 2>/dev/null
                    return ;;
            esac
            if [[ "$cur" == -* ]]; then
                COMPREPLY=( $(compgen -W "--scenario --save-dir --help" -- "$cur") )
            fi
            ;;
    esac
}

complete -F _analyze_autopilot analyze_autopilot.py
