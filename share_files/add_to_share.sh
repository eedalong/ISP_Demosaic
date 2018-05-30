basepath=$(cd `dirname $0`; pwd)
cd /home/yuanxl/dalong_research/
git checkout share_modules

cp $basepath/dalong_layers.py /home/yuanxl/dalong_research/share_files/
cp $basepath/dalong_loss.py   /home/yuanxl/dalong_research/share_files/
cp $basepath/models.py        /home/yuanxl/dalong_research/share_files/
git add share_files/

