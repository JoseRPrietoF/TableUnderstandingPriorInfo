
gnuplot -persist <<-EOFMarker
plot "$1/loss.txt" u 1:2 with linespoints ps 0 title "loss_cv1", \
"$1/loss.txt" u 1:3 with linespoints ps 0 title "val_loss_cv1", \
"$1/loss.txt" u 1:4 with linespoints ps 0 title "loss_cv2", \
"$1/loss.txt" u 1:5 with linespoints ps 0 title "val_loss_cv2", \
"$1/loss.txt" u 1:6 with linespoints ps 0 title "loss_cv3", \
"$1/loss.txt" u 1:7 with linespoints ps 0 title "val_loss_cv3", \
"$1/loss.txt" u 1:8 with linespoints ps 0 title "loss_cv4", \
"$1/loss.txt" u 1:9 with linespoints ps 0 title "val_loss_cv4"
EOFMarker