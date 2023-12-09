(ns my-stuff.core
  (:gen-class))

(defn calc [f x y]
  (f x y))

(defn add
  "I add two numbers"
  [x y]
  (+ x y))

(defn subtract
  "I subtract two numbers"
  [x y]
  (- x y))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println (calc add 1 2)))
