import subprocess
import sys

def run_shell_script(step, size):
    str_param1 = f"./Benchmark/Outputs/nbodyCpuOutput{size}.h5"
    str_param2 = f"./Benchmark/Outputs/nbody{step}Output{size}.h5"
    command = ['sh', './compare.sh', str_param1, str_param2]
    print("running: ", ''.join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    
    print(result.stdout)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare.py <#_of_test>")
        sys.exit(1)
    
    step = int(sys.argv[1])


    for size in [4096]:#, 8192, 12288, 16384, 20480, 24576, 28672, 32768, 36864, 40960, 45056, 49152, 53248, 57344, 61440, 65536, 69632, 73728, 77824, 81920]:
      run_shell_script(step, size)