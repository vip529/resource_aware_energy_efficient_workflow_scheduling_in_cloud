# Resource-Aware Energy Efficient Workflow Scheduling in Cloud Infrastructure

import math as m
import numpy as np
from collections import defaultdict
import random as r
import matplotlib.pyplot as plt



class WorkFlow: 
    def __init__(self,tasks): 
        self.work_fl = defaultdict(list) 
        self.T = tasks 
    def addDependency(self,u,v,w): 
        self.work_fl[u].append((v,w)) 
        
    def predecessor(self,t):
        predecessor_t = list()
        for k in self.work_fl.keys():
            for j in self.work_fl[k]:
                if(j[0] == t):
                    predecessor_t.append((k,j[1]))
        return predecessor_t
                
        
    def successor(self,t):
        succesor_t = list()
        if(self.work_fl[t]):
            for task,com_cost in self.work_fl[t]:
                succesor_t.append((task,com_cost))
        return succesor_t
            
            
        
    def topologicalSortUtil(self,v,visited,stack): 
  
        visited[v] = True
        if v in self.work_fl.keys(): 
            for node,weight in self.work_fl[v]: 
                if visited[node] == False: 
                    self.topologicalSortUtil(node,visited,stack) 

        stack.insert(0,v) 
        
    def reversetopologicalSort(self): 

        visited = [False]*self.T 
        stack =list() 
   
        for i in range(self.T): 
            if visited[i] == False: 
                self.topologicalSortUtil(i,visited,stack) 
        list_tasks = stack[:-1]
        list_tasks.reverse()
        return list_tasks


#The average expected task runtime
def avg_etr(t,vm_count,ETR):
    b = np.array(ETR)
    b = b.transpose()
    avg_etr_value = 0
    for j in range(len(b[t])):
        avg_etr_value = avg_etr_value + b[t][j]
    avg_etr_value = avg_etr_value / vm_count
    return avg_etr_value
        


def EST(wf,t_i,vm,avail,finish_time,assign):
    pr = wf.predecessor(t_i+1)
    oth = list()
    if(pr):
        for t in pr:
            if(assign[t[0]-1] == vm):
                oth.append(finish_time[t[0]-1])
            else:
                oth.append(finish_time[t[0]-1] + t[1])
    else:
        oth.append(0)
    est = max(avail[vm], max(oth))
    return est

#Earliest finish time
def EFT(ETR,t_i,vm,est):
    etr = ETR[vm][t_i]    
    eft = etr + est
    return eft


#the utilization of VMj

def vm_util(b,etc,vm,makespan):
    sum_etr = 0
    for i in range(len(etc[vm])):
        if(b[vm][i] == 1):
            sum_etr = sum_etr + etc[vm][i]
    utilization = sum_etr/makespan
    return utilization



#it is actually summation of vm_cost*(sum of duration of tasks ran on this vm)
# def cost(schedule,etr,vm_cost):
#     p = list()
#     for i in schedule:
#         if(i[1] not in p):
#             p.append(i[1])
#     vm_tasks = [(list(),p[i]) for i in range(len(p))]
            
#     for t in schedule:
#         for v_t,v in vm_tasks:
#             if(t[1] == v):
#                 v_t.append(etr[v][t[0]])
    
#     vm_mk_c = [0 for i in range(len(p))]
#     for ft,v in vm_tasks:
#         vm_mk_c[v] = sum(ft)*vm_cost[v]

#     total_cost = sum(vm_mk_c)
    
#     return total_cost

    
    


def cost_2(schedule,etr,b,vm_count):
    

    sigma = 1
    tau = 1
    beta = 1
    V_cbase = 0.49396
    
    p = list()
    for i in schedule:
        if(i[1] not in p):
            p.append(i[1])
    vm_tasks = [(list(),p[i]) for i in range(len(p))]
            
    for t in schedule:
        for v_t,v in vm_tasks:
            if(t[1] == v):
                v_t.append(etr[v][t[0]])

    
    vm_mk_c = [0 for i in range(len(p))]
    for ft,v in vm_tasks:
        vm_mk_c[v] = sum(ft)
        

    slowest_cpu = max(vm_mk_c)
    
 
    total_cost = 0
    for i in range(vm_count):
        for j in range(len(etr[i])): 
            cost = sigma * (etr[i][j] / tau) * V_cbase *  m.exp(vm_mk_c[i]/slowest_cpu)
            total_cost = total_cost + (cost * b[i][j])
    me = beta * total_cost
    return me


#the sum of energy consumed during the idle state and the busy state of all the VMs
#the total energy of the schedule for a given application over k active VMs
# SL = schedule length = makespan (both paper have same formula)
# busyTime = Actual working time (AWT)

def energy(etc,b,makespan,vm_count):
    
    total_energy = 0
    for i in range(vm_count):
        awt_j = 0
        for j in range(len(etc[i])):
            if(b[i][j] == 1):
                awt_j = awt_j + etc[i][j]
        total_energy= total_energy + (awt_j * 70) + (makespan - awt_j) * 20
    return total_energy



def energy_vfr(vm_count,etc,b,s_volt,freq):
    c = 1
    total_energy = 0
    for i in range(len(etc[0])):
        e_vm_i = 0
        for j in range(vm_count):
            if(b[j][i] == 1):
                e_vm_i = e_vm_i + (c * m.pow(s_volt[j],2) * freq[j] * etc[j][i])
        total_energy  = total_energy + e_vm_i
            
                                           
    return total_energy



def load_balance(vm_count,etc,b,schedule,finish):
    
    awtj = [0 for i in range(vm_count)]
    
    for i in range(vm_count):
        for j in range(len(etc[i])):
            if(b[i][j] == 1):
                awtj[i] = awtj[i] + etc[i][j]
            
    avg_awt = sum(awtj)/vm_count
    
    sd = 0 
    
    for i in range(vm_count):
        sd = sd + m.pow((awtj[i] - avg_awt),2)
    
    sd = m.sqrt(sd/vm_count)
    
    lbf = (abs(avg_awt - sd) / avg_awt) * 100
    
    p = list()
    
    for i in schedule:
        if(i[1] not in p):
            p.append(i[1])

    vm_tasks = [(list(),p[i]) for i in range(len(p))]


    for t in schedule:
        for v_t,v in vm_tasks:
            if(t[1] == v):
                v_t.append(finish[t[0]])
    vm_rtl = [0 for i in range(len(p))]

    for ft,v in vm_tasks:
        vm_rtl[v] = max(ft)
    
    avg_rtl = sum(vm_rtl)/len(vm_tasks)

    lb = 0
    for i in range(vm_count):
        lb = m.pow(avg_rtl - vm_rtl[i] , 2)
    
    lb =m.sqrt(lb/len(vm_rtl))
    
    return (lb,lbf)
         

def speedupt(etc,makespan,vm_count):

    
    eet = [0 for i in range(vm_count)]
    
    for i in range(vm_count):
        for j in range(len(etc[i])):
            eet[i] = eet[i] + etc[i][j]
    
    speedup = max(eet)/(makespan)

    
    return speedup
    

#The algorithm initially generates a schedule using Rankup of all the tasks based on the availability of VMs in the VM pool

def rank_up(wf,task_count,avg_Etr_array):
    revTopList =  wf.reversetopologicalSort()
    rank_up = [0 for i in range(task_count)]
    
    for t in revTopList:
        maxLen = 0
        if(wf.successor(t)):
            for ty in wf.successor(t):
                dt = ty[1]
                if (rank_up[ty[0]-1]+dt >= maxLen):               
                    maxLen= rank_up[ty[0]-1] + dt
        else:
            maxLen = 0 
            
        rank_up[t-1] = maxLen + avg_Etr_array[t-1]
        rank_up_tuple = [(i,val) for i,val in enumerate(rank_up)]
    return rank_up_tuple
            


#resource-aware energy-efficient workflow scheduling

def REWS(wf,ETR,tasks_count,vm_count,deadline,budget,vm_cost):
    
    
    s_volt = [r.uniform(0.7,2) for i in range(vm_count)]
    freq = [r.uniform(0.1,1) for i in range(vm_count)]
    print('voltage: '+ str(s_volt))
    print('oper freq: ' + str(freq))

    new_schedule = list()
    finish_time = [0 for i in range(tasks_count)] 
    start_time = [0 for i in range(tasks_count)]
    makespan = 0
    vm_utilization = 0
    lb_rtl = 0
    lbf = 0
    energy_bi = 0
    energy_vf = 0
    speedup = 0
    T_cost = 0
    
    while(True):   
        avail_vm = [0 for i in range(vm_count)]
        
        task_assign = [-1 for i in range(tasks_count)]
        
        avg_Etr_array = [avg_etr(t,vm_count,ETR) for t in range(tasks_count)]
        
        rankup = rank_up(wf,tasks_count,avg_Etr_array)
        revrank_sorted_task = sorted(rankup, key = lambda kv:(-kv[1], kv[0]))
        
        q_g = [t[0] for t in revrank_sorted_task]
    
        b = [[0 for j in range(tasks_count)] for i in range(vm_count)]
        
        t_i = -1
        
        while(q_g):
            t_i = q_g.pop(0)
            eft = list()
            
            for vm in range(vm_count):
                
                estv = EST(wf,t_i,vm,avail_vm,finish_time,task_assign) 
                eftv = EFT(ETR,t_i,vm,estv)
                eft.append(eftv)
            
            assign = ( t_i ,eft.index(min(eft)) )
            vm_index = eft.index(min(eft)) 
            t_i_start = EST(wf,t_i,vm_index,avail_vm,finish_time,task_assign)
            start_time[t_i] = t_i_start
            finish_time[t_i]=EFT(ETR,t_i,vm_index,t_i_start)
            avail_vm[vm_index]= finish_time[t_i]
            task_assign[t_i]=vm_index;
            new_schedule.append(assign)
        
        for i in range(vm_count):
            for j in range(tasks_count):
                if(task_assign[j] == i ):
                    b[i][j] = 1
                else:
                    b[i][j] = 0
         
        
        makespan = finish_time[t_i]
        vm_util_array = [round(100*vm_util(b,ETR,vm,makespan),2) for vm in range(vm_count)]
        
        vm_utilization = vm_util_array
        lb_rtl,lbf = load_balance(vm_count,ETR,b,new_schedule,finish_time)
        energy_bi = energy(ETR,b,makespan,vm_count)
        energy_vf = energy_vfr(vm_count,ETR,b,s_volt,freq)
        speedup = speedupt(ETR,makespan,vm_count)
        #T_cost = cost(new_schedule,ETR,vm_cost)
        T_cost = cost_2(new_schedule,ETR,b,vm_count)
        
        if (makespan < deadline and T_cost < budget):
            if(vm_count > 1):
                min_util_VM = vm_util_array.index(min(vm_util_array))
                ETR.pop(min_util_VM)
                vm_cost.pop(min_util_VM)
                vm_count = vm_count - 1
                s_volt.pop(min_util_VM)
                freq.pop(min_util_VM)
                new_schedule.clear()
                finish_time = [0 for i in range(tasks_count)]
                start_time = [0 for i in range(tasks_count)]
                makespan = 0
                vm_utilization = 0
                lb_rtl = 0
                lbf = 0
                energy_bi = 0
                energy_vf = 0
                speedup = 0
                T_cost = 0
            else:
                break
        else:
            break
   
    return (new_schedule,start_time,finish_time,makespan,vm_utilization,energy_bi,round(energy_vf,2),round(lb_rtl,2),round(lbf,2),round(speedup,2),T_cost)
    


def plotSchedule(schedule,tasks_count):
    start = schedule[1]
    finish = schedule[2]
    tasks_duration  = [(start[i],finish[i]-start[i],i) for i in range(tasks_count)]

    p = list()
    
    for i in schedule[0]:
        if(i[1] not in p):
            p.append(i[1])

    vm_tasks = [(list(),p[i]) for i in range(len(p))]

    for t in schedule[0]:
        for v_t,v in vm_tasks:
            if(t[1] == v):
                v_t.append(tasks_duration[t[0]])
                
    vm_task_final = [list() for i in range(len(p))] 
    
    for tsk in vm_tasks:
        vm_task_final[tsk[1]] = tsk[0]
     
    
    vm_task_range = [list() for i in range(len(p))]
    
    for i,tsk in enumerate(vm_task_final):
        for j in tsk:
            vm_task_range[i].append((j[0],j[1]))

    fig,ax = plt.subplots()
    
    for i in range(len(vm_task_final)):
        ax.broken_barh( vm_task_range[i], (15+ (15*i)-3, 6), animated = True, edgecolor = 'black' ,
                       facecolors = [ (r.random(), r.random(), r.random())for i in range(len(vm_task_final[i])) ], 
                       label = [ax.text( t[0]+(t[1]//2), 15+ (15*i)-7 , s='t'+str(t[2])) for t in vm_task_final[i]])  
        
    ax.set_ylim(0, (15 * len(vm_task_final) +10) )
    
    ax.set_xlim(0, schedule[3]+20)
    
    ax.set_xlabel('runtime')
    
    yticks = [(15+15*i) for i in range(len(vm_task_final))]
    
    yticks_label = ['vm'+str(i) for i in range(len(vm_task_final)) ]
    
    ax.set_yticks(yticks)
    
    ax.set_yticklabels(yticks_label)
    
    #ax.grid(True)
    plt.title("Energy and Resource aware Workflow Scheduling\n makespan = "+str(schedule[3]))
    
    


def input_dataset(filename):
    with open('Datasets/'+filename,'r') as fh:
        tasks_count, vm_count = fh.readline().strip().split(' ')
        tasks_count = int(tasks_count)
        vm_count = int(vm_count)
        
        ETR = [[0 for i in range(tasks_count)] for j in range(vm_count)]
        for i in range(tasks_count):
            t = list(map(int,fh.readline()[:-1].strip().split(' ')))
            for j in range(len(t)):
                ETR[j][i] = t[j]
        
        mac_comp = [[0 for i in range(vm_count)] for j in range(vm_count)]
        for i in range(vm_count):
            t = list(map(int,fh.readline()[:-1].strip().split(' ')))
            for j in range(len(t)):
                mac_comp[i][j] = t[j]
        
        wfg = WorkFlow(tasks_count+1)
        
        for i in range(tasks_count):
            ts = list(map(int,fh.readline()[:-1].strip().split(' ')))
            for j in range(len(ts)):
                if(ts[j] != -1):
                    if(ts[j] > 1):
                        wfg.addDependency(i+1,j+1,ts[j])
                    else:
                        wfg.addDependency(i+1,j+1,0)
        return (wfg,ETR,tasks_count,vm_count)
                            

def mainf(filename):
    
    
    wf,ETR,tasks_count,vm_count = input_dataset(filename)
    
    return (wf,ETR,tasks_count,vm_count)
    


if __name__ == "__main__":
    
    #keep your datasets in a folder named 'Datasets' inside the current directory
    
    wf,ETR,tasks_count,vm_count = mainf('Montage398.txt')
    deadline = 82
    budget = 84
    vm_cost = [0 for i in range(vm_count)]
    
    schedule = REWS(wf,ETR,tasks_count,vm_count,deadline,budget,vm_cost)
    print("final schedule: " + str(schedule[0]))
    print("makespan: " + str(schedule[3]))
    print("vm_utilization: " + str(schedule[4]))
    print("energy_bi: " + str(schedule[5]))
    print("energy_vf: " + str(schedule[6]))
    print("lb_rtl: " + str(schedule[7]))
    print("lbf: " + str(schedule[8]))
    print("speedup: " + str(schedule[9]))
    print("Total_cost: " + str(schedule[10]))
    plotSchedule(schedule,tasks_count)





