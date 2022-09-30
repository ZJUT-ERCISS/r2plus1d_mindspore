echo "================================================================================================"
echo "Please run the script as: "
echo "bash train_distribute.sh [PROJECT_PATH] [DATA_PATH]"
echo "For example: bash train_distribute.sh /home/r2plus1d /home/publicfile/kinetics-400"
echo "================================================================================================"
set -e
if [ $# -lt 2 ]; then
  echo "Usage: bash train_distribute.sh [PROJECT_PATH] [DATA_PATH]"
exit 1
fi

PYTHON_PATH=$1
DATA_PATH=$2

export PATH=/usr/local/openmpi-4.0.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/openmpi-4.0.3/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHON_PATH
mpirun -n 4 --allow-run-as-root python $PYTHON_PATH/src/example/r2plus1d_kinetics400_train.py --run_distribute True \
    --data_url $DATA_PATH >  train_distributed.log 2>&1

