basepath=$(cd `dirname $0`; pwd)
cd /home/yuanxl/dalong_research/
git checkout share_modules
cp $basepath/dalong_layers.py /home/yuanxl/dalong_research/share_files/
cp $basepath/dalong_loss.py   /home/yuanxl/dalong_research/share_files/
cp $basepath/dalong_models.py  /home/yuanxl/dalong_research/share_files/
cp $basepath/add_to_share.sh  /home/yuanxl/dalong_research/share_files/
git add share_files/
git commit -m 'modify share_files'
git push origin share_modules
cd $basepath
