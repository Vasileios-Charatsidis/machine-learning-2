all:
	@make *.tex

%.pdf: %.tex
	TEXINPUTS="..:" pdflatex $<

clean:
	rm *.log *.aux *.out

