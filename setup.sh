#!/bin/bash
cd ${0%/*}
if [ ! -d "ncbi-blast-2.12.0+/" ]; then
	echo "blast 2.12 will be downloaded..."
	wget -q "ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.12.0/ncbi-blast-2.12.0+-x64-linux.tar.gz"
		if [ $? -eq 0 ] 
		then 
			echo "OK" 
		else 
			echo "Could not download."
			exit 1
		fi
	mkdir blast
	tar -xvf ncbi-blast-2.12.0+-x64-linux.tar.gz
	rm ncbi-blast-2.12.0+-x64-linux.tar.gz
fi
if [ ! -d "venv38/" ]; then
	echo "Python3 virtual environment will be created..."
	python3 -m venv --copies venv38/
	source venv38/bin/activate
	pip install -q -r requirements.txt
	echo "OK"
fi

echo "done!"
