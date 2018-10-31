all:
	cd src && $(MAKE) compile
all-local:
	cd src && $(MAKE) compile-local
clean:
	cd src && $(MAKE) clean
debug:
	cd src && $(MAKE) debug
debug-local:
	cd src && $(MAKE) debug-local
memtest:
	cd src && $(MAKE) memtest
memtest-local:
	cd src && $(MAKE) memtest-local