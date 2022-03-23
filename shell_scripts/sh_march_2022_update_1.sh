# python -W ignore loading_data.py -s 125504.0,114712.0,111436.0,214845.0,211109.0,208515.0,215690.0,111359.0,101930.0,214995.0,102349.0,108910.0,210832.0,203270.0,124042.0,104767.0,134901.0,215452.0,206043.0,102213.0,100964.0,123316.0,212463.0,214458.0,102113.0,208407.0,211749.0,107602.0,125232.0,204478.0,215368.0,213110.0,207846.0,106069.0,109459.0,151769.0,106617.0,214623.0,140014.0,124167.0,213747.0,210784.0,213432.0,214806.0,111668.0,207013.0,101121.0,215631.0,166805.0,213073.0,108516.0,120453.0,103406.0,104117.0,6646.0,213668.0,127218.0,214242.0,144597.0,139787.0,213777.0,215935.0,112934.0,189279.0,108637.0,109491.0,124033.0,214039.0,214605.0,121802.0,102550.0,203947.0,102879.0,137522.0,216404.0,105337.0,127546.0,214022.0,129490.0,166318.0,189359.0,215477.0,142765.0,214321.0,208475.0,125736.0,103915.0,108735.0,109667.0,216422.0  -b 2021 -e 2021 -o march_2022_update_part_1_10
# python -W ignore loading_data.py -s 108794.0,101571.0,215646.0,136197.0,110952.0,214930.0,214407.0,108764.0,109423.0,208881.0,212405.0,214416.0,105231.0,112285.0,112865.0,147806.0,210336.0,143129.0,216383.0,142792.0,208247.0,103134.0,214672.0,108856.0,209823.0,213551.0,107483.0,115984.0,213788.0,212944.0,109486.0,208031.0,216360.0,214835.0,205553.0,112894.0,116545.0,213823.0,188656.0,125455.0,109999.0,113944.0,215287.0,215430.0,210835.0,120916.0,216148.0,212401.0,102968.0,204268.0,208828.0,210338.0,205680.0,104468.0,215700.0,137684.0,212771.0,102164.0,112783.0,210640.0,107315.0,213126.0,104633.0,214730.0,214792.0,208063.0,215158.0,111504.0,111079.0,216066.0,134685.0,209303.0,211751.0,216589.0,107601.0,205180.0,104530.0,211721.0,189702.0,203894.0,213771.0,123099.0,210612.0,104508.0,207609.0,166686.0,125181.0,107528.0,108793.0,101210.0  -b 2021 -e 2021 -o march_2022_update_part_1_11
# python -W ignore loading_data.py -s 143025.0,135069.0,102771.0,154126.0,139976.0,214340.0,106226.0,106778.0,215675.0,211190.0,216305.0,145634.0,209301.0,125179.0,124129.0,101920.0,101209.0,102110.0,213512.0,152904.0,104743.0,205640.0,121206.0,145991.0,213159.0,216218.0,106137.0,101138.0,212063.0,214795.0,108448.0,102896.0,101397.0,154313.0,115225.0,208292.0,213773.0,155239.0,101519.0,207039.0,101697.0,212330.0,109903.0,213198.0,213802.0,216438.0,112233.0,100991.0,214252.0,148228.0,161220.0,103943.0,209299.0,214913.0,215666.0,212564.0,215010.0,209964.0,210847.0,207442.0,109343.0,213642.0,112079.0,207764.0,109615.0,212009.0,109492.0,211550.0,111133.0,116527.0,106884.0,204758.0,213860.0,209320.0,214482.0,215222.0,148257.0,113430.0,203574.0,113103.0,205113.0,137351.0,204238.0,110169.0,110244.0,105997.0,121524.0,214181.0,107341.0,216412.0  -b 2021 -e 2021 -o march_2022_update_part_1_14

# /Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia -p 4 fit_smiles.jl march_2022_update_part_1_8
# /Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia -p 4 est_parameters.jl march_2022_update_part_1_8
# echo 'Done with group 8/15' >> tracking_file_march_2022_update.txt

# /Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia -p 4 fit_smiles.jl march_2022_update_part_1_9
# /Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia -p 4 est_parameters.jl march_2022_update_part_1_9
# echo 'Done with group 9/15' >> tracking_file_march_2022_update.txt

# /Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia -p 4 fit_smiles.jl march_2022_update_part_1_10
# /Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia -p 4 est_parameters.jl march_2022_update_part_1_10
# echo 'Done with group 10/15' >> tracking_file_march_2022_update.txt

/Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia -p 4 fit_smiles.jl march_2022_update_part_1_11
/Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia -p 4 est_parameters.jl march_2022_update_part_1_11
echo 'Done with group 11/15' >> tracking_file_march_2022_update.txt

/Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia -p 4 fit_smiles.jl march_2022_update_part_1_12
/Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia -p 4 est_parameters.jl march_2022_update_part_1_12
echo 'Done with group 12/15' >> tracking_file_march_2022_update.txt

/Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia -p 4 fit_smiles.jl march_2022_update_part_1_13
/Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia -p 4 est_parameters.jl march_2022_update_part_1_13
echo 'Done with group 13/15' >> tracking_file_march_2022_update.txt

/Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia -p 4 fit_smiles.jl march_2022_update_part_1_14
/Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia -p 4 est_parameters.jl march_2022_update_part_1_14
echo 'Done with group 14/15' >> tracking_file_march_2022_update.txt

/Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia -p 4 fit_smiles.jl march_2022_update_part_1_15
/Applications/Julia-1.7.app/Contents/Resources/julia/bin/julia -p 4 est_parameters.jl march_2022_update_part_1_15
echo 'Done with group 15/15' >> tracking_file_march_2022_update.txt

