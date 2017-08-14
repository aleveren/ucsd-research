NAME=presentation

TEXPATH=/Library/TeX/texbin

.PHONY: clean deepclean

all: clean
	${TEXPATH}/pdflatex ${NAME}
	${TEXPATH}/pdflatex ${NAME}
	${TEXPATH}/bibtex ${NAME}
	${TEXPATH}/pdflatex ${NAME}
	${TEXPATH}/pdflatex ${NAME}

clean:
	rm -f *.bbl *.blg *.log *.out *.aux *.nav *.snm *.toc

deepclean: clean
	rm -f ${NAME}.pdf
