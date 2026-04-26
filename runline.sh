uv run nsys profile -t cuda,nvtx -s none -o report_laptop python cs336_systems/benchmark.py run_preset --warmup 5 --n_steps 3 --on_laptop False
uv run nsys profile -t cuda,nvtx -s none -o report_laptop python cs336_systems/benchmark.py run_preset --warmup 5 --n_steps 3 --on_laptop True

for config in medium xl; do for cl in 256 512 2048; do uv run nsys profile -t cuda,nvtx -s none -o report_cl_${cl}_config_${config} python cs336_systems/benchmark.py run_preset --warmup 5 --n_steps 3 --on_laptop False --context_length $cl --config $config; done; done
