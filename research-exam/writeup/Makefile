NAME=Leverentz_research_exam

TEXPATH=/Library/TeX/texbin

.PHONY: clean deepclean

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
