## declare an array variable
declare -a arr=("posteriors_01" "posteriors_12" "posteriors_02" "cov_y" "cov_z" "dist_z" "xevo" "ytest" "y_dist" "yevo_0" "yevo_1" "yevo_2")

## now loop through the above array
for i in "${arr[@]}"
do
   echo "---> making movie for $i data ..."
   convert -resize 512x512 -delay 20 /home/hunter.gabbard/public_html/CBC/cINNamon/gausian_results/multipar/gpu7/${i}_*.png -loop 1 /home/hunter.gabbard/public_html/CBC/cINNamon/gausian_results/multipar/gpu7/animations/${i}.gif
   # or do whatever with individual element of the array
done
