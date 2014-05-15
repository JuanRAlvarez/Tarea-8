all: x2vsalpha.pdf
x2vsalpha.pdf : graficas.py
	python graficas.py

graficas.py: valores.dat

valores.dat: LotkaVolterra.py
	python LotkaVolterra.py

clean:
	rm *.pdf valores.dat