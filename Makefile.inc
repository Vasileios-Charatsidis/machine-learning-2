all:
	@make *.pdf

%.pdf: %.tex
	TEXINPUTS="..:" pdflatex $<

clean:
	rm *.log *.aux *.out

