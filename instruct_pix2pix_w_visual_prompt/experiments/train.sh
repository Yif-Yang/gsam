jobname=$1
cmd_mode=$2
dataset_names=${3:-'instruct,celebahq,celebaedge,deepfashion,ade20k,lowlevelpair,styleclip'}
machine_rank=${NODE_RANK}
main_process_port=${MASTER_PORT}
main_process_ip=${MASTER_IP}


export PATH=${HOME}/.local/bin:$PATH

${HOME}/.local/bin/wandb login 56d149bd571b8312fbca5e3802d7859909ea00c1

run_cmd="python"
if [[ ${cmd_mode} == 'python' ]]; then
  unset WORLD_SIZE
  unset NODE_RANK
  WORLD_SIZE=1
  NODE_RANK=0
fi

if [[ ${cmd_mode} == 'para' ]]; then
  unset WORLD_SIZE
  unset NODE_RANK
  WORLD_SIZE=1
  NODE_RANK=0
  run_cmd="accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8"
fi

if [[ ${cmd_mode} == 'more_para' ]]; then
  run_cmd="accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=24 --num_machines=3 \
        --machine_rank=${machine_rank} --main_process_port=${main_process_port} --main_process_ip=${main_process_ip}"
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
  INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/train/'
  jobname=${jobname}_${dataset_names}
  OUTPUT_DIR="Outputs/${jobname}"

  unet_checkpoint_path="Outputs/instruct_pix2pix_wp_inpainting512_vclip_wbotk_eva2_selfattn/checkpoint-30000/unet/unet/diffusion_pytorch_model.bin"
  vision_enc_checkpoint_path="Outputs/instruct_pix2pix_wp_inpainting512_vclip_wbotk_eva2_selfattn/checkpoint-30000/unet/unet/eva2_vision_prompt_encoder.pth"
  copy_ckpt ${unet_checkpoint_path}
  copy_ckpt ${vision_enc_checkpoint_path}

  ${run_cmd} examples/instruct_pix2pix/train_instruct_pix2pix_vp.py \
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
    --report_to=wandb
fi



#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512_vclip_wbotk_eva2_selfattn_mix_ds' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/train/'
#  OUTPUT_DIR="Outputs/${jobname}"
#  unet_checkpoint_path='checkpoints/instruct_pix2pix_wp_inpainting512/checkpoint-30000/unet/diffusion_pytorch_model.bin'
#
#  ${run_cmd} examples/instruct_pix2pix/train_instruct_pix2pix_vp.py \
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
#    --top_k=-16 \
#    --use_eva2 \
#    --wandb_jobname=${jobname} \
#    --cross_or_self=1 \
#    --cross_attention_dim1=768 \
#    --dataset_names 'instruct' 'celebahqedge' 'celebahq' 'deepfashion' \
#    --report_to=wandb
#fi


#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512_vclip_wbotk_eva2_selfattn' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/train/'
#  OUTPUT_DIR="Outputs/${jobname}"
#  unet_checkpoint_path='checkpoints/instruct_pix2pix_wp_inpainting512/checkpoint-30000/unet/diffusion_pytorch_model.bin'
#
#  ${run_cmd} examples/instruct_pix2pix/train_instruct_pix2pix_vp.py \
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
#    --top_k=-16 \
#    --use_eva2 \
#    --wandb_jobname=${jobname} \
#    --cross_or_self=1 \
#    --cross_attention_dim1=768 \
#    --report_to=wandb
#fi



#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512_vclip_wbotk_eva2_selfattn_ade20k' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/train/'
#  unet_checkpoint_path="Outputs/instruct_pix2pix_wp_inpainting512_vclip_wbotk_eva2_selfattn/checkpoint-30000/unet/unet/diffusion_pytorch_model.bin"
#  vision_enc_checkpoint_path="Outputs/instruct_pix2pix_wp_inpainting512_vclip_wbotk_eva2_selfattn/checkpoint-30000/unet/unet/eva2_vision_prompt_encoder.pth"
#  copy_ckpt ${unet_checkpoint_path}
#  copy_ckpt ${vision_enc_checkpoint_path}
#  OUTPUT_DIR="Outputs/${jobname}"
#
#  ${run_cmd} examples/instruct_pix2pix/train_instruct_pix2pix_vp.py \
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
#    --dataset_names 'ade20k' \
#    --report_to=wandb
#fi
#
#
#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512_vclip_wbotk_eva2_selfattn_deepfashion' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/train/'
#  unet_checkpoint_path="Outputs/instruct_pix2pix_wp_inpainting512_vclip_wbotk_eva2_selfattn/checkpoint-30000/unet/unet/diffusion_pytorch_model.bin"
#  vision_enc_checkpoint_path="Outputs/instruct_pix2pix_wp_inpainting512_vclip_wbotk_eva2_selfattn/checkpoint-30000/unet/unet/eva2_vision_prompt_encoder.pth"
#  copy_ckpt ${unet_checkpoint_path}
#  copy_ckpt ${vision_enc_checkpoint_path}
#  OUTPUT_DIR="Outputs/${jobname}"
#
#  ${run_cmd} examples/instruct_pix2pix/train_instruct_pix2pix_vp.py \
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
#    --dataset_names 'deepfashion' \
#    --report_to=wandb
#fi




#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512_vclip_wbotk_eva2' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/train/'
#  OUTPUT_DIR="Outputs/${jobname}"
#  unet_checkpoint_path='checkpoints/instruct_pix2pix_wp_inpainting512/checkpoint-30000/unet/diffusion_pytorch_model.bin'
#
#  ${run_cmd} examples/instruct_pix2pix/train_instruct_pix2pix_vp.py \
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
#    --top_k=-16 \
#    --use_eva2 \
#    --wandb_jobname=${jobname} \
#    --cross_attention_dim1=768 \
#    --report_to=wandb
#fi


#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512_both_wbotk_eva2' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/train/'
#  OUTPUT_DIR="Outputs/${jobname}"
#  unet_checkpoint_path='checkpoints/instruct_pix2pix_wp_inpainting512/checkpoint-30000/unet/diffusion_pytorch_model.bin'
#
#  ${run_cmd} examples/instruct_pix2pix/train_instruct_pix2pix_vp.py \
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
#    --top_k=-16 \
#    --use_eva2 \
#    --wandb_jobname=${jobname} \
#    --cross_attention_dim1=768 \
#    --report_to=wandb
#fi


#
#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512_vclip_wbotk_eva2_selfattn_body_face_ds' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/train/'
#  OUTPUT_DIR="Outputs/${jobname}"
#  unet_checkpoint_path='checkpoints/instruct_pix2pix_wp_inpainting512/checkpoint-30000/unet/diffusion_pytorch_model.bin'
#
#  ${run_cmd} examples/instruct_pix2pix/train_instruct_pix2pix_vp.py \
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
#    --top_k=-16 \
#    --use_eva2 \
#    --wandb_jobname=${jobname} \
#    --cross_or_self=1 \
#    --cross_attention_dim1=768 \
#    --dataset_names 'celebahqedge,celebahq,deepfashion' \
#    --report_to=wandb
#fi
#
#


## first very slowly finetuning unet
#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512_vclip_eva2' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/train/'
#  OUTPUT_DIR="Outputs/${jobname}"
#  unet_checkpoint_path='checkpoints/instruct_pix2pix_wp_inpainting512/checkpoint-30000/unet/diffusion_pytorch_model.bin'
#
#  ${run_cmd} examples/instruct_pix2pix/train_instruct_pix2pix_vp.py \
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
#    --train_inpainting=True \
#    --vision_enc_learning_rate=1e-7 \
#    --task_name='instruct_pix2pix_inpainting' \
#    --unet_checkpoint_path=${unet_checkpoint_path} \
#    --use_vision_instruction \
#    --use_eva2 \
#    --report_to=wandb \
#    --wandb_jobname=${jobname}
#fi
#
#
#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512_both_eva2' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/train/'
#  OUTPUT_DIR="Outputs/${jobname}"
#  unet_checkpoint_path='checkpoints/instruct_pix2pix_wp_inpainting512/checkpoint-30000/unet/diffusion_pytorch_model.bin'
#
#  ${run_cmd} examples/instruct_pix2pix/train_instruct_pix2pix_vp.py \
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
#    --use_eva2 \
#    --report_to=wandb \
#    --wandb_jobname=${jobname}
#fi


## first very slowly finetuning unet
#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512_vclip_resume_ckpt3000_both' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/train/'
#  OUTPUT_DIR="Outputs/${jobname}"
#  unet_checkpoint_path='checkpoints/instruct_pix2pix_wp_inpainting512/checkpoint-30000/unet/diffusion_pytorch_model.bin'
#
#  ${run_cmd} examples/instruct_pix2pix/train_instruct_pix2pix_vp.py \
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
#    --report_to=wandb
#fi



#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/train/'
#  OUTPUT_DIR="Outputs/${jobname}"
#
#  ${run_cmd} examples/instruct_pix2pix/train_instruct_pix2pix_vp.py \
#    --pretrained_model_name_or_path=$MODEL_NAME \
#    --output_dir=$OUTPUT_DIR \
#    --resolution=512 --random_flip \
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
#    --train_inpainting=True \
#    --task_name='instruct_pix2pix_inpainting' \
#    --report_to=wandb
#fi


# first very slowly finetuning unet
#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512_vclip' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/train/'
#  OUTPUT_DIR="Outputs/${jobname}"
#  unet_checkpoint_path='checkpoints/instruct_pix2pix_wp_inpainting512/checkpoint-30000/unet/diffusion_pytorch_model.bin'
#
#  ${run_cmd} examples/instruct_pix2pix/train_instruct_pix2pix_vp.py \
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
#    --train_inpainting=True \
#    --vision_enc_learning_rate=1e-7 \
#    --task_name='instruct_pix2pix_inpainting' \
#    --unet_checkpoint_path=${unet_checkpoint_path} \
#    --use_vision_instruction \
#    --report_to=wandb
#fi


#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512_both_wtopk_eva2' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/train/'
#  OUTPUT_DIR="Outputs/${jobname}"
#  unet_checkpoint_path='checkpoints/instruct_pix2pix_wp_inpainting512/checkpoint-30000/unet/diffusion_pytorch_model.bin'
#
#  ${run_cmd} examples/instruct_pix2pix/train_instruct_pix2pix_vp.py \
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
#    --top_k=16 \
#    --use_eva2 \
#    --wandb_jobname=${jobname} \
#    --cross_attention_dim1=768 \
#    --report_to=wandb
#fi
#
#
#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512_vclip_wtopk_eva2' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/train/'
#  OUTPUT_DIR="Outputs/${jobname}"
#  unet_checkpoint_path='checkpoints/instruct_pix2pix_wp_inpainting512/checkpoint-30000/unet/diffusion_pytorch_model.bin'
#
#  ${run_cmd} examples/instruct_pix2pix/train_instruct_pix2pix_vp.py \
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
#    --top_k=16 \
#    --use_eva2 \
#    --wandb_jobname=${jobname} \
#    --cross_attention_dim1=768 \
#    --report_to=wandb
#fi


# NOTE: celebahqedge dataset removed, whose input is not cool
#ckpt_path='checkpoints/instruct_pix2pix_wp_inpainting512/checkpoint-30000/unet/diffusion_pytorch_model.bin'
#if [[ -f ${ckpt_path} ]]; then
#  echo ${ckpt_path} 'exists'
#else
#  echo ${ckpt_path} 'does not exist, downloading'
#  mkdir -p checkpoints/instruct_pix2pix_wp_inpainting512/
#  ~/azcopy copy "https://ussclowpriv100data.blob.core.windows.net/v-yashengsun/instruct_pix2pix_w_vp/checkpoints/instruct_pix2pix_wp_inpainting512/checkpoint-30000.tar?sv=2021-10-04&st=2023-02-02T13%3A21%3A08Z&se=2024-05-03T13%3A21%3A00Z&sr=c&sp=racwdxltf&sig=vV5fXzUKmYaOdfapdzpk82lkU5eKwMYuP14qjD4akGs%3D" checkpoints/instruct_pix2pix_wp_inpainting512/checkpoint-30000.tar
#  cd checkpoints/instruct_pix2pix_wp_inpainting512/
#  tar xvf checkpoint-30000.tar
#  cd ../../
#fi


#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting512_rpe' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/train/'
#  OUTPUT_DIR="Outputs/${jobname}"
#
#  ${run_cmd} examples/instruct_pix2pix/train_instruct_pix2pix_vp.py \
#    --pretrained_model_name_or_path=$MODEL_NAME \
#    --output_dir=$OUTPUT_DIR \
#    --resolution=512 --random_flip \
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
#    --train_inpainting=True \
#    --task_name='instruct_pix2pix_inpainting' \
#    --is_rpe \
#    --report_to=wandb
#fi


# env test
#if [[ ${jobname} == 'origin_instruct_pix2pix' ]]; then
#  export MODEL_NAME="runwayml/stable-diffusion-v1-5"
#  export DATASET_ID="fusing/instructpix2pix-1000-samples"
##  DATASET_ID="timbrooks/instructpix2pix-clip-filtered"
##  original_image_column='original_image'
##    --enable_xformers_memory_efficient_attention \
#  ${run_cmd} examples/instruct_pix2pix/train_instruct_pix2pix.py \
#    --pretrained_model_name_or_path=$MODEL_NAME \
#    --dataset_name=$DATASET_ID \
#    --resolution=256 --random_flip \
#    --gradient_accumulation_steps=4 \
#    --train_batch_size=12 --gradient_checkpointing \
#    --num_train_epochs=500 \
#    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
#    --learning_rate=1e-06 --max_grad_norm=1 --lr_warmup_steps=0 \
#    --conditioning_dropout_prob=0.05 \
#    --mixed_precision=fp16 \
#    --scale_lr \
#    --seed=42 \
#    --val_image_url="https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png" \
#    --validation_prompt="make the mountains snowy" #\
##    --report_to=wandb
#fi


###### use v1-5 base model without inpainting setting
#if [[ ${jobname} == 'instruct_pix2pix_wp' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-v1-5"
#  INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/train/'
#  OUTPUT_DIR="Outputs/${jobname}"
#
#  ${run_cmd} examples/instruct_pix2pix/train_instruct_pix2pix_vp.py \
#    --pretrained_model_name_or_path=$MODEL_NAME \
#    --output_dir=$OUTPUT_DIR \
#    --resolution=256 --random_flip \
#    --gradient_accumulation_steps=4 \
#    --train_batch_size=8 --gradient_checkpointing \
#    --num_train_epochs=500 \
#    --checkpointing_steps=5000 --checkpoints_total_limit=1 \
#    --learning_rate=1e-06 --max_grad_norm=1 --lr_warmup_steps=0 \
#    --conditioning_dropout_prob=0.05 \
#    --mixed_precision=fp16 \
#    --scale_lr \
#    --seed=42 \
#    --val_image_url="https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png" \
#    --validation_prompt="make the mountains snowy" \
#    --placeholder_token="<icl>" \
#    --learnable_property="object" \
#    --instance_data_dir=$INSTANCE_DIR \
#    --train_inpainting=True \
#    --report_to=wandb
#fi



#if [[ ${jobname} == 'instruct_pix2pix_wp_inpainting' ]]; then
#  MODEL_NAME="runwayml/stable-diffusion-inpainting"
#  INSTANCE_DIR='/data/yashengsun/local_storage/instruct-pix2pix/train/'
#  OUTPUT_DIR="Outputs/${jobname}"
#
#  ${run_cmd} examples/instruct_pix2pix/train_instruct_pix2pix_vp.py \
#    --pretrained_model_name_or_path=$MODEL_NAME \
#    --output_dir=$OUTPUT_DIR \
#    --resolution=256 --random_flip \
#    --gradient_accumulation_steps=4 \
#    --train_batch_size=8 --gradient_checkpointing \
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
#    --report_to=wandb
#fi
