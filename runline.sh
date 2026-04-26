uv run nsys profile -t cuda,nvtx -s none -o report_laptop python cs336_systems/benchmark.py run_preset --warmup 5 --n_steps 3 --on_laptop False
uv run nsys profile -t cuda,nvtx -s none -o report_laptop python cs336_systems/benchmark.py run_preset --warmup 5 --n_steps 3 --on_laptop True
