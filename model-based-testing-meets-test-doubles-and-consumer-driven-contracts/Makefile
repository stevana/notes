SOURCE=note

all:
	@pandoc --standalone --from markdown --to html \
	-o $(SOURCE).html $(SOURCE).md

clean:
	rm $(SOURCE).html
