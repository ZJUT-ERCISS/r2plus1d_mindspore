echo "================================================================================================"
echo "Please run the script as: "
echo "bash eval_standalone.sh [PROJECT_PATH] [DATA_PATH] [MODEL_PATH]"
echo "For example: bash eval_standalone.sh /home/r2plus1d /home/publicfile/kinetics-400 r2plus1d_kinetics400.ckpt"
echo "================================================================================================"
set -e
if [ $# -lt 2 ]; then
  echo "Usage: bash eval_standalone.sh [PROJECT_PATH] [DATA_PATH] [MODEL_PATH]"
exit 1
fi

PYTHON_PATH=$1
DATA_PATH=$2
MODEL_PATH=$3

export PYTHONPATH=$PYTHON_PATH
python $PYTHON_PATH/src/example/r2plus1d_kinetics400_eval.py --data_url $DATA_PATH \
    --pretrained_model $MODEL_PATH >  eval_result.log 2>&1

