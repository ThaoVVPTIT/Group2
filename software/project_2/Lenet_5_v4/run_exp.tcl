open_project "D:/THUC TAP/Project_vivado/cnn_accelerator/cnn_accelerator.xpr"
add_files -fileset sim_1 "D:/THUC TAP/New folder/tb_experiment.v"
set_property top tb_experiment [get_filesets sim_1]
update_compile_order -fileset sim_1
launch_simulation
run all
close_project
