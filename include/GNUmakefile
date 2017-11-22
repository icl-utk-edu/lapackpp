CXX = g++
CXXFLAGS = -Wconversion -I../../blaspp/include

.PHONY: test_headers clean docs

# --------------------
# compile headers to verify self-sufficiency
header_gch = ${addprefix gch/, ${addsuffix .gch, ${wildcard *.h *.hh}}}

test_headers: ${header_gch}

gch/%.hh.gch: %.hh | gch
	${CXX} ${CXXFLAGS} -c -o $@ $<

gch/%.h.gch: %.h | gch
	${CXX} ${CXXFLAGS} -c -o $@ $<

gch:
	mkdir $@

clean:
	-${RM} gch/*.gch

# --------------------
docs:
	cd .. && ${MAKE} $@
