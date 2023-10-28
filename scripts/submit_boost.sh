SWEEP_ID="31ise655"
BOOST_NUM=-1

sbatch --job-name BOOST-${SWEEP_ID} --cpus-per-task 10 --mem 64G submit_CPU_job.sh \
    "conda activate py310" \
    "python3 boost.py --sweep_id ${SWEEP_ID} --boost_num ${BOOST_NUM}"