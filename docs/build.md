# CPU only
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLUNARA_ENABLE_CUDA=OFF
cmake --build build -j
ctest --test-dir build 
```

# CUDA enabled. No GPU
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLUNARA_ENABLE_CUDA=ON -DLUNARA_ENABLE_GPUTESTS=OFF
cmake --build build -j
ctest --test-dir build 
```

# GPU cloud box
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLUNARA_ENABLE_CUDA=ON -DLUNARA_ENABLE_GPUTESTS=ON
cmake --build build -j
ctest --test-dir build
```
