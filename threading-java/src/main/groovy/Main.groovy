static void main(String[] args) {
    def array = []
    // insert 20 random numbers into the array
    for (int i = 0; i < 20; i++) {
        array << (int)(Math.random() * 100)
    }

    // print the array
    println array

    // divide the array into 4 parts
    def threads = []
    for (int i = 0; i < 4; i++) {
        def start = i * 5
        def end = start + 5
        threads << new Thread({
            array[start..end] = array[start..end].sort()
        })
    }

    // start the threads
    threads.each { it.start() }

    // wait for the threads to finish
    threads.each { it.join() }

    // print the array
    println array
}