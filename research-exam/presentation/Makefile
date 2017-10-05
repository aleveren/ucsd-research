NAME=presentation

TEXPATH=/Library/TeX/texbin

.PHONY: clean deepclean clean-slides clean-handout

all: slides handout

slides: clean-slides
	${TEXPATH}/pdflatex ${NAME}-slides
	${TEXPATH}/pdflatex ${NAME}-slides
	${TEXPATH}/bibtex ${NAME}-slides
	${TEXPATH}/pdflatex ${NAME}-slides
	${TEXPATH}/pdflatex ${NAME}-slides

handout: clean-handout
	${TEXPATH}/pdflatex ${NAME}-handout
	${TEXPATH}/pdflatex ${NAME}-handout
	${TEXPATH}/bibtex ${NAME}-handout
	${TEXPATH}/pdflatex ${NAME}-handout
	${TEXPATH}/pdflatex ${NAME}-handout

clean-slides:
	rm -f *-slides.{bbl,blg,log,out,aux,nav,snm,toc}

clean-handout:
	rm -f *-handout.{bbl,blg,log,out,aux,nav,snm,toc}

clean: clean-slides clean-handout

deepclean: clean
	rm -f ${NAME}*.pdf
