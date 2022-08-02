* Prefix sum: http://www.adms-conf.org/2020-camera-ready/ADMS20_05.pdf
  * https://www.youtube.com/watch?v=pYkaUQDS5W8

accumulate + prefix sum seems to be the fastest
scalar performance drop after 200M elements
lock contention with atomics
multithread with shift is only reasonably faster at low (<4) thread counts
both optimized multithreaded approaches perform roughly the same

# TODO
Prefix sum primitive
Scatter primitive
