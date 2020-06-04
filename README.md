# Resource-Aware-Energy-Efficient-Workflow-Scheduling-in-Cloud-Infrastructure
Implementation in python for the paper:

M. S. Kumar, I. Gupta and P. K. Jana, "Resource-Aware Energy Efficient Workflow Scheduling in Cloud Infrastructure," 2018 International Conference on Advances in Computing, Communications and Informatics (ICACCI), Bangalore, 2018, pp. 293-299, doi: 10.1109/ICACCI.2018.8554707.

* [Link to Paper](https://doi.org/10.1109/ICACCI.2018.8554707)

* ##### for speedup, EET = max{total execution time if all task run on only vm[i] for i in range(vm_count)}
* ##### to use *cost* function, give(set) vm_cost array explicitly
* ##### otherwise *cost_2* function will calculate cost
* ##### for more optimum result(i.e. less energy consumption and more vm utilisation), tune budget and deadline parameters
* ##### in *evergy_vf* function, I have taken random values for operating frequency and voltage, take a fixed value. 
* ##### all the dataset should be in 'Dataset' folder
* ##### dataset contain number of task and number of vms in first line, then ETR matrix with [task_count * vm_count] dimension,
* ##### then it contain 2-D communication cost array(adjacency matrix) to tell about task dependency for DAG creation.

* ### Demo
    * ![Example Task](/imgs/example_task_problem.jpg)

    * ![Optimized Schedule of tasks](/imgs/example_scheduling_result.jpg)

#### Additional metrics used:

* ![cost calculation](/imgs/cost_calculation.jpeg)

* ![energy consumption BusyIdle](/imgs/energy_consumption_bi.jpeg)

* ![energy consumption VolatageFrequency](/imgs/energy_voltagefreq.jpeg)

* ![load balancing factor](/imgs/load_balancing_factor.jpeg)

* ![load balancing RTL](/imgs/load_balancing.jpeg)

* ![vm utilisation](/imgs/vm_utilisation.jpeg)

* ![speedup](/imgs/speed_up.jpeg)




