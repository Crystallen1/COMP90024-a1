#!/bin/bash

#SBATCH --job-name=test_1node_8core
#SBATCH --nodes=1
#SBATCH --ntasks=8

#SBATCH --mail-user=yunchen6@student.unimelb.edu.au
#SBATCH --mail-type=ALL

# 生成带时间戳的输出文件名（格式：YYYYMMDD_HHMMSS）
timestamp=$(date +%Y%m%d_%H%M%S)
output_file="result_1node_8core_${timestamp}.out"

# 加载模块
module load Python/3.10.4
module load mpi4py/3.1.4

# 执行并输出到带时间戳的文件
/usr/bin/time -v mpirun -n 8 python main_parallel_read_MPI.py &> "$output_file"
