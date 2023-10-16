# wandb sweep --project sweep_parameters_icdm scripts/config.yml
AGENT_NAME=$1
END=2
for ((i=1;i<=$END;++i)); do
    sbatch --job-name sweep_parameters_icdm_${i} --cpus-per-task 5 submit_1-A100_job.sh \
    "conda activate py310" \
    "wandb agent ${AGENT_NAME}"
done
