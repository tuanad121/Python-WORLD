#!/usr/bin/env bash
rm -rf Python-Wrapper-for-World-Vocoder
git clone https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder.git
cd Python-Wrapper-for-World-Vocoder
git submodule update --init
python setup.py build_ext --inplace && echo "SUCCESS! You can copy `ls pyworld/*.so` into your working folder to use."
