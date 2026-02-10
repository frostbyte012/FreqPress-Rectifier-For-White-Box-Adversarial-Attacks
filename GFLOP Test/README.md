# GFLOP Calculation using FreqPress

## Overview

This guide provides instructions for calculating the Gigaflops Per Operation (GFLOP) metric for FreqPress algorithms using performance profiling tools. The results demonstrate the floating-point operation efficiency of different filter implementations.

## Prerequisites

### System Requirements
- Linux-based system (tested on Intel processors)
- Python 3.x installed
- `perf` tool installed for performance statistics collection
- Root/sudo access for certain operations

### Installation

1. **Install Linux Tools** (for `perf` command):
   ```bash
   sudo apt install linux-tools-common linux-tools-generic linux-tools-$(uname -r)
   ```

2. **Verify Installation**:
   ```bash
   perf --version
   ```

## Running GFLOP Calculations

### Method 1: Using perf stat with Floating-Point Events

Execute the FreqPress script with performance monitoring to track floating-point arithmetic operations:

```bash
perf stat -e fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.scalar_single python FreqPress_gflop.py
```

**Event Explanations:**
- `fp_arith_inst_retired.scalar_double`: Counts retired scalar double-precision floating-point operations
- `fp_arith_inst_retired.scalar_single`: Counts retired scalar single-precision floating-point operations

### Method 2: Direct Python Execution

For baseline testing without performance metrics:

```bash
python FreqPress_gflop.py
```

### Method 3: Using GFLOPS_Marked_Code_Verification.ipynb

A comprehensive Jupyter notebook approach for detailed GFLOP verification with marked code sections and computational complexity analysis.

**Features:**
- Interactive code cells for step-by-step verification
- Marked computational sections for granular GFLOP calculation
- Complexity breakdowns for different algorithms (JPEG, WebP, Butterworth filters, etc.)
- Manual FLOPs estimation based on algorithm structure
- Comparative analysis between different compression and filtering techniques

**Running the Notebook:**

1. **Open in Jupyter**:
   ```bash
   jupyter notebook GFLOPS_Marked_Code_Verification.ipynb
   ```

2. **Run in VS Code** (with Jupyter extension):
   - Open the file in VS Code
   - Select Python kernel
   - Execute cells sequentially

3. **Run in Google Colab**:
   - Upload the notebook to Google Colab
   - Execute cells with GPU/TPU acceleration if needed

**Notebook Structure:**

The notebook contains verified complexity analysis for:
- **JPEG Compression**: Color conversion, DCT, quantization, entropy coding
- **Total Variation Minimization (TVM)**: Gradient computation, optimization iterations
- **WebP Compression**: Encoder complexity analysis
- **Butterworth Filters**: Filter coefficient calculation and application costs
- **FreqPress Combined**: Overall pipeline GFLOP estimation

**Advantages of Notebook Method:**
- Manual FLOP counting for algorithm verification
- Independent of hardware-specific perf events
- Works across different CPU architectures
- Provides detailed breakdowns of computational costs
- Reproducible and documentable results

**Sample Output from Notebook:**
```
JPEG Complexity Breakdown (224×224×3):
  Color conversion: X,XXX,XXX ops
  DCT:              X,XXX,XXX ops
  Quantization:     X,XXX,XXX ops
  Entropy coding:   X,XXX,XXX ops
  Total GFLOPs:     X.XXXXXX
```


### FreqPress Combined GFLOP
**Result: 0.01044 GFLOP**

## Alternative GFLOP Verification Techniques

Beyond the primary perf-based and notebook methods, several complementary approaches can verify GFLOP calculations:

### 1. **Intel VTune Profiler**
Advanced performance analysis with detailed event sampling:

```bash
vtune -collect performance-snapshot -r results_dir python FreqPress_gflop.py
vtune -report summary -r results_dir
```

**Benefits:**
- GUI-based visualization
- More granular event tracking
- Better handling of modern CPUs
- Supports multiple performance metrics simultaneously

### 2. **PAPI (Performance API)**
Hardware counter interface for portable performance measurements:

```python
import pypapi
from pypapi import papi_high

# Initialize PAPI
papi_high.start_counters([papi_high.Events.FP_OPS])

# Your code here
result = process_freqpress_image()

# Stop and retrieve results
counters = papi_high.stop_counters()
print(f"FP Operations: {counters[0]}")
print(f"GFLOPS: {counters[0] / 1e9 / execution_time}")
```

**Installation:**
```bash
sudo apt install libpapi-dev
pip install py-papi
```

### 3. **CPU Clock Cycle Analysis**
Estimate GFLOP from execution time and CPU frequency:

```python
import time
import subprocess

# Get CPU frequency
cpu_freq = subprocess.check_output("lscpu | grep 'CPU max MHz'", shell=True).decode()

# Measure execution time
start = time.perf_counter()
result = process_freqpress_image()
elapsed = time.perf_counter() - start

# Estimate GFLOP
clock_cycles = float(cpu_freq.split()[-1]) * 1e6 * elapsed
# Assuming N operations per cycle
estimated_gflop = (clock_cycles * ops_per_cycle) / 1e9
```

### 4. **OProfile - System-Wide Profiling**
Kernel-level profiling for comprehensive performance analysis:

```bash
sudo oprofiled --event=FP_OPS,CPU_CLK_UNHALTED
operf python FreqPress_gflop.py
opreport --long-samples
```

**Installation:**
```bash
sudo apt install oprofile oprofile-gui
```

### 5. **AMD CodeXL / AMD uProf**
For AMD processors:

```bash
AMDuProfCLI collect app -e FP_OPS -o results python FreqPress_gflop.py
AMDuProfCLI report -d results
```

### 6. **Manual Complexity Analysis (Notebook Method)**
Theoretical GFLOP calculation based on algorithm structure:

```python
def calculate_theoretical_gflops(image_height, image_width, algorithm_type):
    """
    Calculate theoretical GFLOP based on algorithm complexity
    """
    # Count basic operations (multiplications, additions, divisions)
    total_ops = 0
    
    if algorithm_type == "jpeg":
        # RGB to YCbCr: 9 muls + 6 adds per pixel
        total_ops += image_height * image_width * 15
        # DCT: ~64*log2(64) per 8x8 block
        blocks = (image_height // 8) * (image_width // 8) * 3
        total_ops += blocks * 64 * 6  # ~6 bits = log2(64)
        # ... add more operations for quantization, entropy
    
    execution_time = measure_execution_time()  # in seconds
    gflops = (total_ops / 1e9) / execution_time
    return gflops
```

### 7. **LIKWID (Lightweight Performance Tools)**
Lightweight performance analysis tool:

```bash
# Install LIKWID
sudo apt install likwid

# Run with marker API
likwid-perfctr -C 0 -g FLOPS_DP python FreqPress_gflop.py

# Or use timeline collection
likwid-perfctr -c 0 -m -o output.csv python FreqPress_gflop.py
```

## Comparison of Verification Methods

| Method | Accuracy | Ease of Use | Cross-Platform | Real-time Data |
|--------|----------|------------|-----------------|----------------|
| perf stat | High | Easy | Linux/Intel | Yes |
| Notebook (Manual) | Medium | Medium | All | No |
| Intel VTune | Very High | Medium | Windows/Intel | Yes |
| PAPI | High | Medium | Multi-platform | Yes |
| OProfile | High | Hard | Linux | Yes |
| AMD uProf | High | Medium | AMD/Windows | Yes |
| LIKWID | Very High | Medium | Linux | Yes |

## Recommended Verification Strategy

For comprehensive GFLOP verification, combine multiple methods:

1. **Primary Method**: perf stat (hardware-accurate, lightweight)
2. **Verification Method**: Notebook-based complexity analysis (cross-platform, detailed)
3. **Advanced Profiling** (optional): VTune or LIKWID for detailed bottleneck analysis
4. **Cross-Architecture Testing**: Use notebook method to verify results on different processors

## Running Multiple Verifications

**Combined Verification Script:**

```bash
#!/bin/bash
echo "=== Method 1: perf stat ==="
perf stat -e fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.scalar_single python FreqPress_gflop.py

echo -e "\n=== Method 2: Jupyter Notebook ==="
jupyter notebook GFLOPS_Marked_Code_Verification.ipynb

echo -e "\n=== Method 3: CPU Frequency Analysis ==="
lscpu | grep "CPU max MHz"
```

## Results Validation

After running multiple GFLOP verification methods:

1. **Compare Results**: All methods should yield similar GFLOP values (±5-10% acceptable variation)
2. **Document Variations**: Record differences and their potential causes
3. **Identify Outliers**: Investigate any methods producing significantly different results
4. **Hardware Confirmation**: Verify processor model and frequency for consistent comparisons


## Processor-Specific Considerations

### Intel Processors
- Uses `fp_arith_inst_retired.scalar_double` and `fp_arith_inst_retired.scalar_single` events
- Supports hardware-level performance monitoring
- Recommended for accurate GFLOP measurements

### AMD Processors
For AMD systems, use alternative events:
```bash
perf stat -e fp_op_retired.all python FreqPress_gflop.py
```

### ARM Processors
For ARM-based systems, check available events:
```bash
perf list | grep floating
```

## Troubleshooting

### Permission Issues
If you encounter permission errors:
```bash
sudo perf stat -e fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.scalar_single python FreqPress_gflop.py
```

### Event Not Found
If the floating-point events are not available on your system:
1. Verify your processor supports these events
2. Check kernel version compatibility (should be 4.0+)
3. Update perf tools if necessary

### No Events Recorded
This may indicate:
- Insufficient permissions
- Unsupported processor architecture
- Outdated perf version

## System Verification Checklist

- [ ] Linux tools installed (`apt install linux-tools-common linux-tools-generic`)
- [ ] `perf` tool accessible and working (`perf --version`)
- [ ] Python environment configured correctly
- [ ] FreqPress_gflop.py file present and executable
- [ ] Sufficient permissions for performance monitoring
- [ ] System idle to minimize interference

## Usage for Other Systems

### Step-by-Step Guide for Users

1. **Check Your Processor**:
   ```bash
   lscpu | grep "Model name"
   ```

2. **Install Required Tools**:
   ```bash
   sudo apt install linux-tools-common linux-tools-generic linux-tools-$(uname -r)
   ```

3. **Verify Available Events** (Intel):
   ```bash
   perf list | grep "fp_arith"
   ```

4. **Run the Script**:
   ```bash
   perf stat -e fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.scalar_single python FreqPress_gflop.py
   ```

5. **Compare Results**: Document your results and compare with the baseline values provided in this guide.

## Performance Metrics Explanation

**GFLOP (Gigaflops)**: Measures billions of floating-point operations per second
- Scalar double: 64-bit floating-point operations
- Scalar single: 32-bit floating-point operations

The combined metric provides a comprehensive view of both precision levels used by FreqPress.

## Notes

- Results may vary based on system load, CPU frequency, and processor model
- For consistent results, run tests multiple times and average the values
- Disable CPU frequency scaling for more reproducible measurements:
  ```bash
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
  ```

## References

- [Linux perf documentation](https://linux.die.net/man/1/perf)
- FreqPress algorithm documentation
- Intel VTune Profiler for advanced analysis
