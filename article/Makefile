SRC=article.tex
DATA=jmlda.sty lit.bib
TARGET=article.pdf

LATEXMK_ARGS=-pdf -bibtex

$(TARGET): $(SRC) $(DATA)
	latexmk $(LATEXMK_ARGS) $(SRC)

clean:
	latexmk -CA

.PHONY: $(TARGET)
