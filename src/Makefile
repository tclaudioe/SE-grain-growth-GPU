clean:
	rm ../SEVC result.err result.out debug.txt timeline.prof || true
compile-local: clean
	optirun nvcc --compiler-options -Wall main.cu -o SEVC -ccbin=g++53
	mv SEVC ../SEVC
compile: clean
	nvcc main.cu -o SEVC
	mv SEVC ../SEVC
send: compile
	qsub submission.sh
debug: clean
	nvcc -g -G -D VDEBUG --compiler-options -Wall -lineinfo main.cu -o SEVC
	mv SEVC ../SEVC
memtest: clean
	nvcc --compiler-options -Wall -lineinfo main.cu -o SEVC
	mv SEVC ../SEVC
debug-local: clean
	optirun nvcc -g -G -D VDEBUG --compiler-options -Wall -O0 -lineinfo -ccbin=g++53 main.cu -o SEVC
	mv SEVC ../SEVC
memtest-local: clean
	nvcc --compiler-options -Wall -lineinfo -ccbin=g++53 main.cu -o SEVC
	mv SEVC ../SEVC
