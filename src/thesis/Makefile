THESIS = csuthesis_main
# TEX, BIB, TEST dir
# modified from sjtu thesis
TEX_DIR = content
BIB_DIR = .
OUTPUT_DIR = output

STUDENT_ID = id
STUDENT_NAME = name
DISCIPLANCE = disciplance
TITLE = title

# Option for latexmk
LATEXMK_OPT_BASE = -xelatex -gg -silent
LATEXMK_OPT = $(LATEXMK_OPT_BASE) -f
LATEXMK_OPT_PVC = $(LATEXMK_OPT_BASE) -pvc

all: $(THESIS).pdf
	mv $(OUTPUT_DIR)/$(THESIS).pdf $(STUDENT_ID)_$(STUDENT_NAME)_$(DISCIPLANCE)_$(TITLE).pdf

.PHONY : all cleanall pvc view wordcount git zip

$(THESIS).pdf : $(THESIS).tex $(TEX_DIR)/*.tex $(BIB_DIR)/*.bib csuthesis.cls Makefile
	-latexmk $(LATEXMK_OPT) $(THESIS) -output-directory=$(OUTPUT_DIR)

pvc :
	latexmk $(LATEXMK_OPT_PVC) $(THESIS)

view : $(THESIS).pdf
#for Linux user
	xdg-open  $<
#for Mac user
#	open $<

wordcount:
	@texcount $(THESIS).tex -inc -ch-only | awk '/total/ {getline; print "纯中文字数\t\t\t:",$$4}'
	@texcount $(THESIS).tex -inc -chinese | awk '/total/ {getline; print "总字数（英文单词 + 中文字）\t:",$$4}'

clean :
	-@latexmk -c -silent 2> /dev/null
	-@rm -f $(TEX_DIR)/*.aux 2> /dev/null || true

cleanall :
	-@latexmk -C -silent 2> /dev/null
	-@rm -f $(TEX_DIR)/*.aux 2> /dev/null || true

zip :
	git archive --format zip --output thesis.zip master
