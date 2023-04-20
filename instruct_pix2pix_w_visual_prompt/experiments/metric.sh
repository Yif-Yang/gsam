jobname=$1


if [[ $jobname == 'metric' ]]; then
  output_path='/data/yashengsun/Proj/instruct_vp_results/0407_selfattn_ckpt30000'
  python examples/instruct_pix2pix/metrics/compute_metrics.py \
      --output_path ${output_path}
fi
