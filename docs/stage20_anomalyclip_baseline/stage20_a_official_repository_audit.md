# Stage 20-A: Official AnomalyCLIP Repository Audit

## Repository

- implementation: `zqhang/AnomalyCLIP`
- local external path: `/root/private_data/third_party/AnomalyCLIP`
- pinned commit: `3911738c0867544f545a076ad78f3f11d9ecbfdf`
- latest commit: `3911738 Update README.md`

## Current project environment

- Python: `3.12.7`
- PyTorch: `2.7.0+cu118`
- GPU: `NVIDIA GeForce RTX 4090`
- VRAM: approximately `24 GB`

## Decision

Do not install AnomalyCLIP dependencies into the existing project environment.

Use a separate Conda environment because the official implementation uses
an older PyTorch stack and includes its own modified CLIP implementation.

## Official requirements

----- requirements.txt begin -----
scikit-image==0.20.0
scikit-learn==1.2.2
scipy==1.9.1
seaborn==0.12.2
timm==0.6.13
torch==2.0.0
torchsummary==1.5.1
torchvision==0.15.1
tqdm==4.65.0
dash-table==5.0.0
----- requirements.txt end -----

## Official test command

----- test.sh begin -----

device=0

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset mvtec \
        --data_path /remote-home/iot_zhouqihang/data/mvdataset --save_path ./results/${base_dir}/zero_shot \
        --checkpoint_path ${save_dir}epoch_15.pth \
         --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
    wait
    done
done


LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_multiscale_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test.py --dataset visa \
        --data_path /remote-home/iot_zhouqihang/data/Visa --save_path ./results/${base_dir}/zero_shot \
        --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
    wait
    done
done
----- test.sh end -----

## Next step

Create an isolated AnomalyCLIP environment, verify model loading, and
generate AD2-four-category metadata without modifying the source dataset.
