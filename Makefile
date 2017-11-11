all: dummy

dummy:
	python setup.py build_ext --inplace
	rm -rf build

clean :
	rm world/*.c world/*.so
	rm -rf build