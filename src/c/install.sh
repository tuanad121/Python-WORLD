rm -Rf World
git clone https://github.com/mmorise/World.git
git clone https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder.git
mv Python-Wrapper-for-World-Vocoder/* World/
cd World
python setup.py build_ext --inplace 
echo "SUCCESS! copy `ls *.so` into your working folder"
cd ..
rm -Rf Python-Wrapper-for-World-Vocoder