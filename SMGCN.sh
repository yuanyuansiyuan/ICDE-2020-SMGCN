#!/bin/bash
mess_dropouts=('[0.0,0.0] ')
adj_types=('norm')
embed_sizes=(64)
lrs=(2e-4)
regs=('[7e-3]')

for mess_dropout in ${mess_dropouts[@]};
do
    for reg in ${regs[@]};
    do
        for adj_type in ${adj_types[@]};
        do
            for embed_size in ${embed_sizes[@]};
            do
                for lr in ${lrs[@]};
                do
                    python SMGCN.py --fusion 'add' --result_index 1  --layer_size [128,256] --mlp_layer_size [256] --dataset Set2Set --gpu_id 1 --regs $reg  --embed_size $embed_size --alg_type 'SMGCN' --adj_type  $adj_type --lr $lr  --save_flag 0 --pretrain 0 --batch_size 1024 --epoch 2000 --verbose 1  --mess_dropout  $mess_dropout;
                done
            done
        done
    done
done

