# edsr_trt 성능 체크
perf_analyzer -m edsr_trt -u localhost:8011 -i grpc --shared-memory system --output-shared-memory-size 25165824 --shape input:1,3,512,512 -t 2

# edsr_jit 성능 체크
perf_analyzer -m edsr_jit -u localhost:8011 -i grpc --shared-memory system --output-shared-memory-size 25165824 --shape input:1,3,512,512 -t 2

# preprocess 성능 체크
perf_analyzer -m preprocess -u localhost:8011 -i grpc --shared-memory system --output-shared-memory-size 25165824 --shape input:512,512,3 -t 2

# ensemble_edsr 성능 체크
perf_analyzer -m ensemble_edsr -u localhost:8011 -i grpc --shared-memory system --output-shared-memory-size 25165824 --shape input:512,512,3 -t 2
