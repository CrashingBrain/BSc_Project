$DATA << EOD
#Task start      end
A     2012-11-01 2012-12-31
B     2013-01-01 2013-03-14
C     2013-03-15 2014-04-30
D     2013-05-01 2013-06-30
E     2013-07-01 2013-08-31
F1    2013-09-01 2013-10-31
F2    2013-09-01 2014-01-17
F3    2013-09-01 2014-01-30
F4    2013-09-01 2014-03-31
G1    2013-11-01 2013-11-27
G2    2013-11-01 2014-01-17
L     2013-11-28 2013-12-19
M     2013-11-28 2014-01-17
N     2013-12-04 2014-03-02
O     2013-12-20 2014-01-17
P     2013-12-20 2014-02-16
Q     2014-01-05 2014-01-13
R     2014-01-18 2014-01-30
S     2014-01-31 2014-03-31
T     2014-03-01 2014-04-28
EOD


set terminal png  transparent enhanced font "arial,10" fontscale 1.0 size 600, 400 
set output 'gantt.2.png'
set border 3 front lt black linewidth 1.000 dashtype solid
set xdata time
set format x "%b\n'%y" timedate
set grid nopolar
set grid xtics nomxtics ytics nomytics noztics nomztics nortics nomrtics \
 nox2tics nomx2tics noy2tics nomy2tics nocbtics nomcbtics
set grid layerdefault   lt 0 linecolor 0 linewidth 0.500,  lt 0 linecolor 0 linewidth 0.500
unset key
set style arrow 1 head back filled linecolor rgb "#56b4e9"  linewidth 1.500 dashtype solid size screen  0.020,15.000,90.000  fixed
set style data lines
set mxtics 4.000000
set xtics border in scale 2,0.5 nomirror norotate  autojustify
set xtics  norangelimit 2.6784e+06
set ytics border in scale 1,0.5 nomirror norotate  autojustify
set ytics  norangelimit 
set ytics   ()
set title "{/=15 Simple Gantt Chart}\n\n{/:Bold Task start and end times in columns 2 and 3}" 
set yrange [ -1.00000 : * ] noreverse nowriteback
T(N) = timecolumn(N,timeformat)
timeformat = "%Y-%m-%d"
OneMonth = 2678400.0
GPFUN_T = "T(N) = timecolumn(N,timeformat)"
DEBUG_TERM_HTIC = 119
DEBUG_TERM_VTIC = 119
## Last datafile plotted: "$DATA"
plot $DATA using (T(2)) : ($0) : (T(3)-T(2)) : (0.0) : yticlabel(1) with vector as 1,      $DATA using (T(2)) : ($0) : 1 with labels right offset -2