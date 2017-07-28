NAME=Leverentz_research_exam

.PHONY: clean deepclean

all: clean
	pdflatex ${NAME}
	pdflatex ${NAME}
	bibtex ${NAME}
	pdflatex ${NAME}
	pdflatex ${NAME}

clean:
	rm -f *.bbl *.blg *.log *.out *.aux

deepclean: clean
	rm -f ${NAME}.pdf
