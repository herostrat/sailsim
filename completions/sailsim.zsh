#compdef sailsim
#
# Zsh completion for sailsim
#
# Install (pick one):
#   1. source completions/sailsim.zsh
#   2. Copy to a directory in your $fpath, e.g.:
#      cp completions/sailsim.zsh ~/.zsh/completions/_sailsim
#      (make sure ~/.zsh/completions is in fpath before compinit)

_sailsim_find_configs_root() {
    local dir="$PWD"
    while [[ "$dir" != "/" ]]; do
        if [[ -d "$dir/configs" ]]; then
            echo "$dir/configs"
            return 0
        fi
        dir="${dir:h}"
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
        for f in "$dir"/*.toml(N); do
            echo "${f:t:r}"
        done
    fi
}

_sailsim() {
    local -a opts
    opts=(
        '--scenario[Path to scenario TOML file]:scenario:_sailsim_complete_scenario'
        '--yacht[Yacht profile name or TOML path]:yacht:_sailsim_complete_yacht'
        '--autopilot[Autopilot profile name or TOML path]:autopilot:_sailsim_complete_autopilot'
        '--output[Path to output CSV file]:output file:_files -g "*.csv"'
        '--save-json[Save recording as JSON file]:json file:_files -g "*.json"'
        '*--view[Launch viewer or load JSON files]:json file:_files -g "*.json"'
        '--quiet[Suppress progress output]'
        '--help[Show help message]'
    )

    _arguments -s $opts
}

_sailsim_complete_scenario() {
    local -a scenario_files
    local configs_root
    configs_root="$(_sailsim_find_configs_root)"

    if [[ -n "$configs_root" && -d "$configs_root/scenarios" ]]; then
        scenario_files=( "$configs_root"/scenarios/*.toml(N) )
        if (( ${#scenario_files} )); then
            _values 'scenario' ${scenario_files[@]} && return
        fi
    fi

    _files -g '*.toml'
}

_sailsim_complete_yacht() {
    local -a profiles
    profiles=( $(_sailsim_profiles yachts) )

    local -a alternatives
    alternatives=()
    if (( ${#profiles} )); then
        alternatives+=( "profiles:yacht profile:(${profiles[*]})" )
    fi
    alternatives+=( "files:TOML file:_files -g '*.toml'" )

    _alternative $alternatives
}

_sailsim_complete_autopilot() {
    local -a profiles
    profiles=( $(_sailsim_profiles autopilots) )

    local -a alternatives
    alternatives=()
    if (( ${#profiles} )); then
        alternatives+=( "profiles:autopilot profile:(${profiles[*]})" )
    fi
    alternatives+=( "files:TOML file:_files -g '*.toml'" )

    _alternative $alternatives
}

compdef _sailsim sailsim
compdef _sailsim 'python -m sailsim'
