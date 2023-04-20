CUDA_VISIBLE_DEVICES=7, bash experiments/test.sh instruct_pix2pix_wp_inpainting512 single 7.5 1.5 3.0 "Outputs/instruct_pix2pix_wp_inpainting512/checkpoint-20000/unet/diffusion_pytorch_model.bin" &
CUDA_VISIBLE_DEVICES=6, bash experiments/test.sh instruct_pix2pix_wp_inpainting512 single 7.5 2.5 3.0 "Outputs/instruct_pix2pix_wp_inpainting512/checkpoint-20000/unet/diffusion_pytorch_model.bin" &
CUDA_VISIBLE_DEVICES=5, bash experiments/test.sh instruct_pix2pix_wp_inpainting512 single 5.5 1.5 3.0 "Outputs/instruct_pix2pix_wp_inpainting512/checkpoint-20000/unet/diffusion_pytorch_model.bin" &
CUDA_VISIBLE_DEVICES=4, bash experiments/test.sh instruct_pix2pix_wp_inpainting512 single 5.5 2.5 3.0 "Outputs/instruct_pix2pix_wp_inpainting512/checkpoint-20000/unet/diffusion_pytorch_model.bin" &
CUDA_VISIBLE_DEVICES=1, bash experiments/test.sh instruct_pix2pix_wp_inpainting512 single 0.0 2.5 3.0 "Outputs/instruct_pix2pix_wp_inpainting512/checkpoint-20000/unet/diffusion_pytorch_model.bin" &

CUDA_VISIBLE_DEVICES=3, bash experiments/test.sh instruct_pix2pix_wp_inpainting512_rpe single 5.5 2.5 3.0 "Outputs/instruct_pix2pix_wp_inpainting512_rpe/checkpoint-20000/unet/diffusion_pytorch_model.bin" &
CUDA_VISIBLE_DEVICES=2, bash experiments/test.sh instruct_pix2pix_wp_inpainting512_rpe single 7.5 1.5 3.0 "Outputs/instruct_pix2pix_wp_inpainting512_rpe/checkpoint-20000/unet/diffusion_pytorch_model.bin" &
CUDA_VISIBLE_DEVICES=0, bash experiments/test.sh instruct_pix2pix_wp_inpainting512_rpe single 0.0 2.5 3.0 "Outputs/instruct_pix2pix_wp_inpainting512_rpe/checkpoint-20000/unet/diffusion_pytorch_model.bin" &
