import os
import subprocess

# パスの設定
current_dir = os.getcwd()
x_file = os.path.join(current_dir, '../data/armadillo-g-y.txt')
y_file = os.path.join(current_dir, '../data/armadillo-g-x.txt')
bcpd_bin = os.path.join(current_dir, '../bcpd' if not os.name == 'nt' else '../win/VisualStudio/x64/Release/BCPD-Win.exe')
fnf = os.path.join(current_dir, '../data/armadillo-g-triangles.txt')

# パラメータの設定
omg = '0.0'
bet = '1.0'
lmd = '50'
gma = '.1'
K = '200'
J = '300'
c = '1e-6'
n = '500'
nrm = 'x'
dwn = 'B,10000,0.02'
tau = '1'

# 実行コマンドの構築
kern = f'geodesic,{tau},{fnf}'
prm1 = f'-w{omg} -b{bet} -l{lmd} -g{gma}'
prm2 = f'-J{J} -K{K} -p -u{nrm} -D{dwn}'
prm3 = f'-c{c} -n{n} -h -r1'
cmd = f'{bcpd_bin} -x{x_file} -y{y_file} {prm1} {prm2} {prm3} -ux -G{kern} -sY'

# コマンドの実行
subprocess.run(cmd, shell=True)
