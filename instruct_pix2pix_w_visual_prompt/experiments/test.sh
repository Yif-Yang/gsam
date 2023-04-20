
### modify the following startup scripts
### bash experiments/test.sh inpainting512_vclip_wbotk_eva2_selfattn_mix 4, instruct 7.5 1.5 3.0 Outputs/inpainting512_vclip_wbotk_eva2_selfattn_mix__deepfashion/checkpoint-5000/unet/unet/diffusion_pytorch_model.bin Outputs/inpainting512_vclip_wbotk_eva2_selfattn_mix__deepfashion/checkpoint-5000/unet/unet/eva2_vision_prompt_encoder.pth
jobname=$1
gpu_device=${2:-0}
#dataset_names=${3:-'instruct,celebahq,celebaedge,deepfashion,ade20k,lowlevelpair,styleclip'}
dataset_names=${3:-'instruct'}

text_guidance_scale=${4:-7.5}
image_guidance_scale=${5:-1.5}
example_guidance_scale=${6:-3.0}

unet_checkpoint_path=${7:-"Outputs/instruct_pix2pix_wp_inpainting512_vclip_wbotk_eva2_selfattn/checkpoint-30000/unet/unet/diffusion_pytorch_model.bin"}
vision_enc_checkpoint_path=${8:-"Outputs/instruct_pix2pix_wp_inpainting512_vclip_wbotk_eva2_selfattn/checkpoint-30000/unet/unet/eva2_vision_prompt_encoder.pth"}
cmd_mode=${9}

# Note that --val_every_num_instance=50 need to be added, copy vision encoder
INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/val/'

unset WORLD_SIZE
unset NODE_RANK
WORLD_SIZE=1
NODE_RANK=0

export PATH=${HOME}/.local/bin:$PATH
export CUDA_VISIBLE_DEVICES=${gpu_device}

run_cmd="python"
if [[ ${cmd_mode} == 'para' ]]; then
  run_cmd="accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8"
fi

function copy_ckpt {
  ckpt_path="$1"
  if [[ -f "${ckpt_path}" ]]; then
    echo ${ckpt_path} 'exists'
  else
    echo ${ckpt_path} 'does not exist, downloading'
    ~/azcopy copy "https://ussclowpriv100data.blob.core.windows.net/v-yashengsun/instruct_pix2pix_w_visual_prompt/"${ckpt_path}"?sv=2021-10-04&st=2023-02-02T13%3A21%3A08Z&se=2024-05-03T13%3A21%3A00Z&sr=c&sp=racwdxltf&sig=vV5fXzUKmYaOdfapdzpk82lkU5eKwMYuP14qjD4akGs%3D" ${ckpt_path}
  fi
}


if [[ ${jobname} == 'inpainting512_vclip_wbotk_eva2_selfattn_mix' ]]; then
  MODEL_NAME="runwayml/stable-diffusion-inpainting"
  copy_ckpt ${unet_checkpoint_path}
  copy_ckpt ${vision_enc_checkpoint_path}
  OUTPUT_DIR="Outputs/test/${jobname}_${dataset_names}"

  ${run_cmd} examples/instruct_pix2pix/inference_instruct_pix2pix_vp.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --output_dir=$OUTPUT_DIR \
    --resolution=512 \
    --gradient_accumulation_steps=4 \
    --train_batch_size=3 --gradient_checkpointing \
    --num_train_epochs=500 \
    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
    --learning_rate=1e-06 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --scale_lr \
    --seed=42 \
    --placeholder_token="<icl>" \
    --learnable_property="object" \
    --instance_data_dir=$INSTANCE_DIR \
    --use_vision_instruction \
    --train_inpainting=True \
    --vision_enc_learning_rate=1e-7 \
    --task_name='instruct_pix2pix_inpainting' \
    --unet_checkpoint_path=${unet_checkpoint_path} \
    --vision_enc_checkpoint_path=${vision_enc_checkpoint_path} \
    --top_k=-16 \
    --use_eva2 \
    --wandb_jobname=${jobname} \
    --cross_or_self=1 \
    --cross_attention_dim1=768 \
    --dataset_names=${dataset_names} \
    --val_every_num_instance=50
fi


#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512_vclip_wbotk_eva2_selfattn' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  unet_checkpoint_path=${7:-"Outputs/instruct_pix2pix_wp_inpainting512_vclip_wbotk_eva2_selfattn/checkpoint-30000/unet/unet/diffusion_pytorch_model.bin"}
#  vision_enc_checkpoint_path=${8:-"Outputs/instruct_pix2pix_wp_inpainting512_vclip_wbotk_eva2_selfattn/checkpoint-30000/unet/unet/eva2_vision_prompt_encoder.pth"}
#  copy_ckpt ${unet_checkpoint_path}
#  copy_ckpt ${vision_enc_checkpoint_path}
#  OUTPUT_DIR="Outputs/test/${jobname}"
#
#  ${run_cmd} examples/instruct_pix2pix/inference_instruct_pix2pix_vp.py \
#    --pretrained_model_name_or_path=$MODEL_NAME \
#    --output_dir=$OUTPUT_DIR \
#    --resolution=512 \
#    --gradient_accumulation_steps=4 \
#    --train_batch_size=3 --gradient_checkpointing \
#    --num_train_epochs=500 \
#    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
#    --learning_rate=1e-06 --max_grad_norm=1 --lr_warmup_steps=0 \
#    --conditioning_dropout_prob=0.05 \
#    --mixed_precision=fp16 \
#    --scale_lr \
#    --seed=42 \
#    --placeholder_token="<icl>" \
#    --learnable_property="object" \
#    --instance_data_dir=$INSTANCE_DIR \
#    --use_vision_instruction \
#    --train_inpainting=True \
#    --vision_enc_learning_rate=1e-7 \
#    --task_name='instruct_pix2pix_inpainting' \
#    --unet_checkpoint_path=${unet_checkpoint_path} \
#    --vision_enc_checkpoint_path=${vision_enc_checkpoint_path} \
#    --top_k=-16 \
#    --use_eva2 \
#    --wandb_jobname=${jobname} \
#    --cross_or_self=1 \
#    --cross_attention_dim1=768 \
#    --val_every_num_instance=50
#fi


#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512_vclip_wtopk_eva2' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  unet_checkpoint_path=${7:-"Outputs/instruct_pix2pix_wp_inpainting512_vclip_wtopk_eva2/checkpoint-15000/unet/unet/diffusion_pytorch_model.bin"}
#  vision_enc_checkpoint_path=${8:-"Outputs/instruct_pix2pix_wp_inpainting512_vclip_wtopk_eva2/checkpoint-15000/unet/unet/eva2_vision_prompt_encoder.pth"}
#  copy_ckpt ${unet_checkpoint_path}
#  copy_ckpt ${vision_enc_checkpoint_path}
#
#  OUTPUT_DIR="Outputs/test/${jobname}"
#
#  ${run_cmd} examples/instruct_pix2pix/inference_instruct_pix2pix_vp.py \
#    --pretrained_model_name_or_path=$MODEL_NAME \
#    --output_dir=$OUTPUT_DIR \
#    --resolution=512 \
#    --gradient_accumulation_steps=4 \
#    --train_batch_size=4 --gradient_checkpointing \
#    --num_train_epochs=500 \
#    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
#    --learning_rate=1e-06 --max_grad_norm=1 --lr_warmup_steps=0 \
#    --conditioning_dropout_prob=0.05 \
#    --mixed_precision=fp16 \
#    --scale_lr \
#    --seed=42 \
#    --placeholder_token="<icl>" \
#    --learnable_property="object" \
#    --instance_data_dir=$INSTANCE_DIR \
#    --use_vision_instruction \
#    --train_inpainting=True \
#    --vision_enc_learning_rate=1e-7 \
#    --task_name='instruct_pix2pix_inpainting' \
#    --unet_checkpoint_path=${unet_checkpoint_path} \
#    --vision_enc_checkpoint_path=${vision_enc_checkpoint_path} \
#    --top_k=16 \
#    --use_eva2 \
#    --wandb_jobname=${jobname} \
#    --val_every_num_instance=50 \
#    --cross_attention_dim1=768
#fi
#
#
#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512_both_wbotk_eva2' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  unet_checkpoint_path=${7:-"Outputs/instruct_pix2pix_wp_inpainting512_both_wtopk_eva2/checkpoint-15000/unet/unet/diffusion_pytorch_model.bin"}
#  vision_enc_checkpoint_path=${8:-"Outputs/instruct_pix2pix_wp_inpainting512_both_wtopk_eva2/checkpoint-15000/unet/unet/eva2_vision_prompt_encoder.pth"}
#  copy_ckpt ${unet_checkpoint_path}
#  copy_ckpt ${vision_enc_checkpoint_path}
#
#  OUTPUT_DIR="Outputs/test/${jobname}"
#
#  ${run_cmd} examples/instruct_pix2pix/inference_instruct_pix2pix_vp.py \
#    --pretrained_model_name_or_path=$MODEL_NAME \
#    --output_dir=$OUTPUT_DIR \
#    --resolution=512 \
#    --gradient_accumulation_steps=4 \
#    --train_batch_size=4 --gradient_checkpointing \
#    --num_train_epochs=500 \
#    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
#    --learning_rate=1e-06 --max_grad_norm=1 --lr_warmup_steps=0 \
#    --conditioning_dropout_prob=0.05 \
#    --mixed_precision=fp16 \
#    --scale_lr \
#    --seed=42 \
#    --placeholder_token="<icl>" \
#    --learnable_property="object" \
#    --instance_data_dir=$INSTANCE_DIR \
#    --use_vision_instruction \
#    --use_language_instruction \
#    --train_inpainting=True \
#    --vision_enc_learning_rate=1e-7 \
#    --task_name='instruct_pix2pix_inpainting' \
#    --unet_checkpoint_path=${unet_checkpoint_path} \
#    --vision_enc_checkpoint_path=${vision_enc_checkpoint_path} \
#    --top_k=-16 \
#    --use_eva2 \
#    --val_every_num_instance=50 \
#    --cross_attention_dim1=768
#fi


#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  OUTPUT_DIR="Outputs/test/${jobname}"
#
#  ${run_cmd} examples/instruct_pix2pix/inference_instruct_pix2pix_vp.py \
#    --pretrained_model_name_or_path=$MODEL_NAME \
#    --output_dir=$OUTPUT_DIR \
#    --resolution=512 --random_flip \
#    --gradient_accumulation_steps=4 \
#    --train_batch_size=1 --gradient_checkpointing \
#    --num_train_epochs=500 \
#    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
#    --learning_rate=1e-06 --max_grad_norm=1 --lr_warmup_steps=0 \
#    --conditioning_dropout_prob=0.05 \
#    --mixed_precision=fp16 \
#    --scale_lr \
#    --seed=42 \
#    --placeholder_token="<icl>" \
#    --learnable_property="object" \
#    --instance_data_dir=$INSTANCE_DIR \
#    --train_inpainting=True \
#    --task_name='instruct_pix2pix_inpainting' \
#    --unet_checkpoint_path=${unet_checkpoint_path} \
#    --vision_enc_checkpoint_path=${vision_enc_checkpoint_path} \
#    --text_guidance_scale=${text_guidance_scale} \
#    --image_guidance_scale=${image_guidance_scale} \
#    --example_guidance_scale=${example_guidance_scale} \
#    --val_every_num_instance=50
#fi


#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512_rpe' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  OUTPUT_DIR="Outputs/test/${jobname}"
#
#  ${run_cmd} examples/instruct_pix2pix/inference_instruct_pix2pix_vp.py \
#    --pretrained_model_name_or_path=$MODEL_NAME \
#    --output_dir=$OUTPUT_DIR \
#    --resolution=512 --random_flip \
#    --gradient_accumulation_steps=4 \
#    --train_batch_size=1 --gradient_checkpointing \
#    --num_train_epochs=500 \
#    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
#    --learning_rate=1e-06 --max_grad_norm=1 --lr_warmup_steps=0 \
#    --conditioning_dropout_prob=0.05 \
#    --mixed_precision=fp16 \
#    --scale_lr \
#    --seed=42 \
#    --placeholder_token="<icl>" \
#    --learnable_property="object" \
#    --instance_data_dir=$INSTANCE_DIR \
#    --train_inpainting=True \
#    --task_name='instruct_pix2pix_inpainting' \
#    --unet_checkpoint_path=${unet_checkpoint_path} \
#    --vision_enc_checkpoint_path=${vision_enc_checkpoint_path} \
#    --text_guidance_scale=${text_guidance_scale} \
#    --image_guidance_scale=${image_guidance_scale} \
#    --example_guidance_scale=${example_guidance_scale} \
#    --val_every_num_instance=50 \
#    --is_rpe
#fi



#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting256' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/val/'
#  OUTPUT_DIR="Outputs/test/${jobname}"
#  checkpoint_path="Outputs/instruct_pix2pix_wp_inpainting/checkpoint-10000/unet/diffusion_pytorch_model.bin"
#
#  ${run_cmd} examples/instruct_pix2pix/inference_instruct_pix2pix_vp.py \
#    --pretrained_model_name_or_path=$MODEL_NAME \
#    --output_dir=$OUTPUT_DIR \
#    --resolution=256 --random_flip \
#    --gradient_accumulation_steps=4 \
#    --train_batch_size=1 --gradient_checkpointing \
#    --num_train_epochs=500 \
#    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
#    --learning_rate=1e-06 --max_grad_norm=1 --lr_warmup_steps=0 \
#    --conditioning_dropout_prob=0.05 \
#    --mixed_precision=fp16 \
#    --scale_lr \
#    --seed=42 \
#    --placeholder_token="<icl>" \
#    --learnable_property="object" \
#    --instance_data_dir=$INSTANCE_DIR \
#    --train_inpainting=True \
#    --task_name='instruct_pix2pix_inpainting' \
#    --checkpoint_path=${checkpoint_path} \
#    --report_to=wandb
#fi


#~/azcopy copy checkpoint-10000.tar "https://ussclowpriv100data.blob.core.windows.net/v-yashengsun/instruct_pix2pix_w_vp/checkpoints/instruct_pix2pix_wp_inpainting512_rpe/checkpoint-10000.tar?sv=2021-10-04&st=2023-02-02T13%3A21%3A08Z&se=2024-05-03T13%3A21%3A00Z&sr=c&sp=racwdxltf&sig=vV5fXzUKmYaOdfapdzpk82lkU5eKwMYuP14qjD4akGs%3D"
#~/azcopy copy checkpoint-20000.tar "https://ussclowpriv100data.blob.core.windows.net/v-yashengsun/instruct_pix2pix_w_vp/checkpoints/instruct_pix2pix_wp_inpainting512_rpe/checkpoint-20000.tar?sv=2021-10-04&st=2023-02-02T13%3A21%3A08Z&se=2024-05-03T13%3A21%3A00Z&sr=c&sp=racwdxltf&sig=vV5fXzUKmYaOdfapdzpk82lkU5eKwMYuP14qjD4akGs%3D"
#~/azcopy copy checkpoint-30000.tar "https://ussclowpriv100data.blob.core.windows.net/v-yashengsun/instruct_pix2pix_w_vp/checkpoints/instruct_pix2pix_wp_inpainting512/checkpoint-30000.tar?sv=2021-10-04&st=2023-02-02T13%3A21%3A08Z&se=2024-05-03T13%3A21%3A00Z&sr=c&sp=racwdxltf&sig=vV5fXzUKmYaOdfapdzpk82lkU5eKwMYuP14qjD4akGs%3D"
#~/azcopy copy "https://ussclowpriv100data.blob.core.windows.net/v-yashengsun/instruct_pix2pix_w_vp/checkpoints/instruct_pix2pix_wp_inpainting512/checkpoint-30000.tar?sv=2021-10-04&st=2023-02-02T13%3A21%3A08Z&se=2024-05-03T13%3A21%3A00Z&sr=c&sp=racwdxltf&sig=vV5fXzUKmYaOdfapdzpk82lkU5eKwMYuP14qjD4akGs%3D" ./