NAME=Leverentz_research_exam

TEXPATH=/Library/TeX/texbin

.PHONY: clean deepclean count

all: clean
	${TEXPATH}/pdflatex ${NAME}
	${TEXPATH}/pdflatex ${NAME}
	${TEXPATH}/bibtex ${NAME}
	${TEXPATH}/pdflatex ${NAME}
	${TEXPATH}/pdflatex ${NAME}

clean:
	rm -f *.bbl *.blg *.log *.out *.aux

deepclean: clean
	rm -f ${NAME}.pdf

count:
	${TEXPATH}/texcount -1 -sum ${NAME}.tex
