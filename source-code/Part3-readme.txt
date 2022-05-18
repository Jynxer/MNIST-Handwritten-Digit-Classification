Example of compiling and running Part 3 source code:

jr263@pc7-023-l:~/Documents/source-code $ javac -cp lib/jblas-1.2.5.jar:minet:. minet/*.java minet/*/*.java *.java
jr263@pc7-023-l:~/Documents/source-code $ java -cp lib/jblas-1.2.5.jar:minet:. Part3 123 data/mnist_train.txt data/mnist_dev.txt data/mnist_test.txt