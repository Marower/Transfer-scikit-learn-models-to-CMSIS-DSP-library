# Artrial fibrilation clasifier on STM32WB55RG
Implementation of artrial fibrylation calsifiers using CMSIS DSP library for ARM<BR/>

## Communication over serial terminal
### Data opperation
In one line (ended with \r\n) send input vector. Values must be coma separated. For example: <BR/>
 <BR/>
In response device will return prediction of evry implemented clasifiers. Response to example input vector looks: <BR/>
 <BR/>
For evry classifier 0 indicate artrial fibrilation, 1 normal - sinus rhytm.  <BR/>
### Commands
A - procede whole dateset store in memory, return results of classification for all clasifiers. <BR/>
B - meassure time of computation of different clasifiers. <BR/>


## Troubleshooting
This application don't works with turn on optimalization...