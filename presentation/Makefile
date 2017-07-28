NAME=presentation

.PHONY: clean deepclean

all: clean
	pdflatex ${NAME}
	pdflatex ${NAME}
	bibtex ${NAME}
	pdflatex ${NAME}
	pdflatex ${NAME}

clean:
	rm -f *.bbl *.blg *.log *.out *.aux *.nav *.snm *.toc

deepclean: clean
	rm -f ${NAME}.pdf
