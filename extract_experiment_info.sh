#!/bin/bash

INPUT_FILE="experiment_run_output.txt"

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: $INPUT_FILE not found!"
    exit 1
fi

extract_experiment_data() {
    local file="$1"
    local in_experiment=0
    local experiment_line=""
    local best_model_lines=()
    local epoch_lines=()          # rolling buffer of last 2 epoch lines
    local last_two_epochs=()
    local early_stop_line=""
    local test_acc_line=""
    local found_early_stop=0
    local found_test_acc=0
    local prev_line=""

    while IFS= read -r line; do
        # Detect experiment start: line contains [n] and "Training" or "supervised"
        if [[ $line =~ \[[0-9]+\] ]] && [[ $line =~ (Training|supervised) ]]; then
            # If we were already in an experiment, print its data
            if [ $in_experiment -eq 1 ]; then
                print_experiment_data
                # Reset for next experiment
                best_model_lines=()
                epoch_lines=()
                last_two_epochs=()
                early_stop_line=""
                test_acc_line=""
                found_early_stop=0
                found_test_acc=0
                experiment_line=""
            fi
            in_experiment=1
            experiment_line="$line"
            epoch_lines=()   # fresh buffer for this experiment
        fi

        # Process lines if inside an experiment
        if [ $in_experiment -eq 1 ]; then
            # Capture epoch lines (keep last 2)
            if [[ $line =~ Epoch ]]; then
                epoch_lines+=("$line")
                if [ ${#epoch_lines[@]} -gt 2 ]; then
                    epoch_lines=("${epoch_lines[@]:1}")   # keep only last 2
                fi
            fi

            # Best model saved: store the epoch line and the save line
            if [[ $line =~ "Saved new best model" ]]; then
                if [ -n "$prev_line" ] && [[ $prev_line =~ Epoch ]]; then
                    best_model_lines+=("$prev_line")
                fi
                best_model_lines+=("$line")
            fi

            # Early stopping trigger
            if [[ $line =~ "Early stopping triggered" ]]; then
                early_stop_line="$line"
                found_early_stop=1
                # Copy the last two epoch lines at this moment
                last_two_epochs=("${epoch_lines[@]}")
            fi

            # Test accuracy (line with "Accuracy:" but not "Val Acc")
            if [[ $line =~ "Accuracy:" ]] && [[ ! $line =~ "Val Acc" ]]; then
                test_acc_line="$line"
                found_test_acc=1
            fi
        fi

        prev_line="$line"
    done < "$file"

    # Print the last experiment if any
    if [ $in_experiment -eq 1 ]; then
        print_experiment_data
    fi
}

print_experiment_data() {
    echo "=========================================="
    echo "EXPERIMENT: $experiment_line"
    echo "------------------------------------------"
    
    if [ ${#best_model_lines[@]} -gt 0 ]; then
        echo "BEST MODEL SAVES:"
        for ((i=0; i<${#best_model_lines[@]}; i++)); do
            echo "  ${best_model_lines[$i]}"
        done
    else
        echo "BEST MODEL SAVES: None found"
    fi
    echo "------------------------------------------"
    
    # Show the two epoch lines right before early stopping
    if [ $found_early_stop -eq 1 ] && [ ${#last_two_epochs[@]} -gt 0 ]; then
        echo "LAST TWO EPOCHS BEFORE EARLY STOPPING:"
        for epoch_line in "${last_two_epochs[@]}"; do
            echo "  $epoch_line"
        done
    else
        echo "LAST TWO EPOCHS BEFORE EARLY STOPPING: Not available"
    fi
    echo "------------------------------------------"
    
    if [ $found_early_stop -eq 1 ]; then
        echo "EARLY STOPPING: $early_stop_line"
    else
        echo "EARLY STOPPING: Not found"
    fi
    echo "------------------------------------------"
    
    if [ $found_test_acc -eq 1 ]; then
        echo "TEST ACCURACY: $test_acc_line"
    else
        echo "TEST ACCURACY: Not found"
    fi
    echo "=========================================="
    echo
}

extract_experiment_data "$INPUT_FILE"
