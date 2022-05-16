python train.py --arch=relu --use_pe=True --batch_size=640000 --exp_name=raw_mlps_pe_800*800_640000
python train.py --arch=relu --use_pe=False --batch_size=640000 --exp_name=raw_mlps_800*800_640000

python train.py --arch=ff --use_pe=True --batch_size=640000 --exp_name=ff_2pi_800*800_640000 --sc=1.
python train.py --arch=ff --use_pe=True --batch_size=640000 --exp_name=ff_20pi_800*800_640000 --sc=10.
python train.py --arch=ff --use_pe=True --batch_size=640000 --exp_name=ff_200pi_800*800_640000 --sc=100.

python train.py --arch=siren --use_pe=False --omega_0=30 --batch_size=640000 --exp_name=siren_30_800*800_640000

python train.py --arch=gaussian --use_pe=True --a=0.1 --batch_size=640000 --exp_name=gau_a0.1_pe_800*800_640000
python train.py --arch=gaussian --use_pe=False --a=0.1 --batch_size=640000 --exp_name=gau_a0.1_800*800_640000

python train.py --arch=quadratic --use_pe=True --a=10 --batch_size=640000 --exp_name=qua_a10_pe_800*800_640000
python train.py --arch=quadratic --use_pe=False --a=10 --batch_size=640000 --exp_name=qua_a10_800*800_640000

python train.py --arch=multi-quadratic --use_pe=True --a=20 --batch_size=640000 --exp_name=multiqua_a20_pe_800*800_640000
python train.py --arch=multi-quadratic --use_pe=False --a=20 --batch_size=640000 --exp_name=multiqua_a20_800*800_640000

python train.py --arch=laplacian --use_pe=True --a=0.1 --batch_size=640000 --exp_name=laplacian_a0.1_pe_800*800_640000
python train.py --arch=laplacian --use_pe=False --a=0.1 --batch_size=640000 --exp_name=laplacian_a0.1_800*800_640000

python train.py --arch=super-gaussian --use_pe=True --a=0.1 --b=2 --batch_size=640000 --exp_name=supgau_a0.1_b2_pe_800*800_640000
python train.py --arch=super-gaussian --use_pe=False --a=0.1 --b=2 --batch_size=640000 --exp_name=supgau_a0.1_b2_800*800_640000

python train.py --arch=expsin --use_pe=True --a=0.1 --batch_size=640000 --exp_name=expsin_a0.1_pe_800*800_640000
python train.py --arch=expsin --use_pe=False --a=0.1 --batch_size=640000 --exp_name=expsin_a0.1_800*800_640000