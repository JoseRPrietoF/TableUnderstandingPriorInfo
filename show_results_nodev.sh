gnuplot -persist <<-EOFMarker
plot "$1/loss.txt" u 1:2 with linespoints ps 0 title "loss_cv1", \
"$1/loss.txt" u 1:3 with linespoints ps 0 title "loss_cv2", \
"$1/loss.txt" u 1:4 with linespoints ps 0 title "loss_cv3", \
"$1/loss.txt" u 1:5 with linespoints ps 0 title "loss_cv4"
EOFMarker