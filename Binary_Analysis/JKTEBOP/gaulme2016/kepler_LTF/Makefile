all: lc.out model.out param.out rvA.out rvB.out Plot.lc.svg Plot.rvA.svg Plot.rvB.svg Plot.ocA.svg Plot.ocB.svg Plot.rv.svg

Plot.lc.svg: plot.lc.gpi lc.out model.out
	gnuplot plot.lc.gpi

Plot.rvA.svg: plot.rvA.gpi rvA.out model.out
	gnuplot plot.rvA.gpi

Plot.rvB.svg: plot.rvB.gpi rvB.out model.out
	gnuplot plot.rvB.gpi

Plot.ocA.svg: plot.ocA.gpi rvA.out
	gnuplot plot.ocA.gpi

Plot.ocB.svg: plot.ocB.gpi rvB.out
	gnuplot plot.ocB.gpi

Plot.rv.svg: plot.rv.gpi rvB.out rvA.out model.out
	gnuplot plot.rv.gpi

lc.out model.out param.out rvA.out rvB.out: jktebop
	./jktebop infile.KEPLER

jktebop: ../../jktebop.f
	gfortran -o kepler_LTF/jktebop ../jktebop.f

.PHONY:clean
clean:
	rm -f *.out *.svg
